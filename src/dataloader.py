import os
import torch
from torch.utils.data import Dataset

class SimpleTokenDataset(Dataset):
    def __init__(
        self,
        token_dir: str,
        model_ids: list[str],
        window_size: int = 256,
        augment: bool = False,
        seed: int = 42,
    ):
        """
        Dataset for Instant-NGP hash-table tokens. For each sliding window of size W (default 256),
        it returns:
          - tokens:   FloatTensor of shape [W, 2]
          - abs_norm: FloatTensor of shape [W, 1], computed as abs_idx / total_tokens
          - p_norm:   FloatTensor of shape [W, 1], computed as pos_in_level / level_sizes[level]
          - levels:   LongTensor of shape [W], containing integer level labels 0..maxLevel

        Level sizes are defined in self.level_sizes and can dynamically adjust to the keys provided.
        """
        super().__init__()  # Initialize the base Dataset class
        self.token_dir   = token_dir      # Directory where token and position files are stored
        self.window_size = window_size    # Number of tokens per window
        self.augment     = augment        # Whether to apply Gaussian noise augmentation to tokens
        self.rng         = torch.Generator().manual_seed(seed)  # Random generator with fixed seed

        # ---------------------------------------------------
        # 1) Define sizes of each level (0..15). Can be extended if more levels are needed.
        # ---------------------------------------------------
        self.level_sizes = {
            0: 4096,
            1: 12168,
            2: 29792,
            3: 79512,
            4: 205384,
            5: 524288,
            6: 524288,
            7: 524288,
            8: 524288,
            9: 524288,
            10: 524288,
            11: 524288,
            12: 524288,
            13: 524288,
            14: 524288,
            15: 524288,
        }
        # Compute total number of tokens across all levels
        self.total_tokens = sum(self.level_sizes.values())

        # ---------------------------------------------------
        # 2) Load all token and position files into memory, once.
        #    Each model_id corresponds to two .pt files: one for tokens and one for positions.
        #    We also store metadata about the number of tokens per model and how many sliding windows fit.
        # ---------------------------------------------------
        self.token_cache = []  # Will hold a list of FloatTensors, each of shape [N_i, 2]
        self.pos_cache   = []  # Will hold a list of FloatTensors, each of shape [N_i, 3]
        self.meta        = []  # Will hold tuples (N_i, max_start_i) for each model

        for mid in model_ids:
            # Construct file paths for tokens and positions
            tpath = os.path.join(token_dir, f"{mid}_tokens.pt")
            ppath = os.path.join(token_dir, f"{mid}_positions.pt")

            # Load token data: FloatTensor of shape [N_i, 2]
            tokens = torch.load(tpath).float()
            # Load position data: FloatTensor of shape [N_i, 3], where each row is [abs_idx, level, pos_in_level]
            positions_raw = torch.load(ppath).float()

            N = tokens.shape[0]  # Number of tokens for this model
            # Compute how many starting indices yield a full window of size `window_size`
            # If N < window_size, at least one window is still valid (we will pad later)
            max_start = max(1, N - window_size + 1)

            # Store loaded data and metadata
            self.token_cache.append(tokens)
            self.pos_cache.append(positions_raw)
            self.meta.append((N, max_start))

    def __len__(self):
        """
        Return the total number of sliding windows across all models.
        This is simply the sum of max_start values for each model in self.meta.
        """
        # Sum up all possible starting positions for each model's token sequence
        return sum(max_start for _, max_start in self.meta)

    def __getitem__(self, idx: int) -> dict:
        """
        Given a global index idx (0 <= idx < total number of windows), map it to a specific
        model i and a start position within that model. Then extract a window of tokens and
        positions [start : start + window_size], optionally pad if at the end, augment if needed,
        normalize positional information, and return a dictionary containing:
            - "tokens":     FloatTensor of shape [W, 2]
            - "abs_norm":   FloatTensor of shape [W, 1]
            - "p_norm":     FloatTensor of shape [W, 1]
            - "levels":     LongTensor of shape [W]
        """
        # ---------------------------------------------------
        # 1) Map global idx to (model index i, window start within that model)
        # ---------------------------------------------------
        cum = 0
        for i, (N, max_start) in enumerate(self.meta):
            # If idx falls within the range for this model
            if idx < cum + max_start:
                start = idx - cum  # Compute the start offset for this specific model
                break
            cum += max_start

        # Retrieve pre-loaded token and position tensors for model i
        tokens   = self.token_cache[i]   # Shape [N_i, 2]
        pos_raw  = self.pos_cache[i]     # Shape [N_i, 3]

        # ---------------------------------------------------
        # 2) Slice out the window: tokens[start : start + window_size]
        # ---------------------------------------------------
        window_tokens    = tokens   [start : start + self.window_size]   # Could be smaller than window_size at end
        window_positions = pos_raw  [start : start + self.window_size]   # Same for positions
        W = window_tokens.shape[0]  # Actual number of tokens in this slice

        # If the slice is shorter than window_size, pad with zeros so that all batches are uniform length
        if W < self.window_size:
            pad_len = self.window_size - W
            # Pad tokens: create a zero tensor of shape [pad_len, 2] and concatenate
            pad_tokens = torch.zeros(pad_len, window_tokens.size(1), dtype=window_tokens.dtype)
            window_tokens = torch.cat([window_tokens, pad_tokens], dim=0)  # Now shape [window_size, 2]
            # Pad positions: create a zero tensor of shape [pad_len, 3] and concatenate
            pad_pos = torch.zeros(pad_len, window_positions.size(1), dtype=window_positions.dtype)
            window_positions = torch.cat([window_positions, pad_pos], dim=0)  # Now shape [window_size, 3]
            W = self.window_size  # Update W to reflect padded length

        # ---------------------------------------------------
        # 3) (Optional) Augmentation: add small Gaussian noise to tokens
        # ---------------------------------------------------
        if self.augment:
            noise = torch.randn_like(window_tokens) * 0.01  # Generate noise with std=0.01
            window_tokens = window_tokens + noise

        # ---------------------------------------------------
        # 4) Normalize position data
        # ---------------------------------------------------
        # window_positions is [W, 3]: columns are [abs_idx, level, pos_in_level]
        abs_idx    = window_positions[:, 0]        # Tensor of shape [W], raw absolute indices
        levels     = window_positions[:, 1].long() # Tensor of shape [W], integer levels 0..15
        pos_in_lvl = window_positions[:, 2]        # Tensor of shape [W], position within its level

        # 4.1) Compute abs_norm = abs_idx / total_tokens → Tensor [W]
        abs_norm = abs_idx / float(self.total_tokens)
        abs_norm = abs_norm.unsqueeze(-1)  # Reshape to [W, 1]

        # 4.2) Compute p_norm = pos_in_lvl / level_size for each entry → Tensor [W]
        p_norm = torch.zeros_like(pos_in_lvl)  # Initialize with zeros
        # Iterate over all defined levels
        for lvl, size in self.level_sizes.items():
            mask = (levels == lvl)  # Boolean mask for tokens at this level
            if mask.any():
                # For entries where level == lvl, divide pos_in_level by size of that level
                p_norm[mask] = pos_in_lvl[mask] / float(size)
        p_norm = p_norm.unsqueeze(-1)  # Reshape to [W, 1]

        # 4.3) levels remain as LongTensor of shape [W]

        # ---------------------------------------------------
        # 5) Return a dictionary with all required fields
        # ---------------------------------------------------
        return {
            "tokens":    window_tokens,  # FloatTensor [W, 2]
            "abs_norm":  abs_norm,       # FloatTensor [W, 1]
            "p_norm":    p_norm,         # FloatTensor [W, 1]
            "levels":    levels          # LongTensor  [W]
        }

