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
        Dataset for Instant-NGP hash-table tokens. Zusätzlich geben wir pro Fenster 
        ein Rotation-Label (0…5) zurück, basierend auf einem vordefinierten Mapping.
        """
        super().__init__()
        self.token_dir   = token_dir
        self.window_size = window_size
        self.augment     = augment
        self.rng         = torch.Generator().manual_seed(seed)

        # ----------------------------------------
        # 0) Mapping "Suffix" → Klassen‐Index (0…5)
        # ----------------------------------------
        # Nach dem zweiten "__" in der Datei (z.B. "asteroid6__base_000_000_000__…"):
        #   "base_000_000_000"   → 0
        #   "compound_090_000_090" → 1
        #   "x_180_000_000"      → 2
        #   "y_000_180_000"      → 3
        #   "z_000_000_120"      → 4
        #   "z_000_000_240"      → 5
        self.rot_labels = {
            "base_000_000_000":      0,
            "compound_090_000_090":  1,
            "x_180_000_000":         2,
            "y_000_180_000":         3,
            "z_000_000_120":         4,
            "z_000_000_240":         5,
        }

        # ---------------------------------------------------
        # 1) Definition der Level‐Größen (kann dynamisch erweitert werden)
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
        self.total_tokens = sum(self.level_sizes.values())

        # ---------------------------------------------------
        # 2) Laden aller Token‐ und Positions‐Files
        # ---------------------------------------------------
        self.token_cache = []
        self.pos_cache   = []
        self.meta        = []  # List of (N_i, max_start_i)
        self.label_cache = []  # List of int (0…5) pro model_id

        for mid in model_ids:
            # mid sieht z. B. so aus: "asteroid6__base_000_000_000__checkpoints__final"
            # Wir extrahieren den Suffix nach dem zweiten "__":
            suffix = mid.split("__")[1]  # z.B. "base_000_000_000"
            if suffix not in self.rot_labels:
                raise KeyError(f"Rotation-Suffix '{suffix}' not found in rot_labels mapping.")
            rot_class = self.rot_labels[suffix]  # Integer 0…5

            tpath = os.path.join(token_dir, f"{mid}_tokens.pt")
            ppath = os.path.join(token_dir, f"{mid}_positions.pt")

            tokens = torch.load(tpath).float()        # [N_i, 2]
            positions_raw = torch.load(ppath).float() # [N_i, 3]

            N = tokens.shape[0]
            max_start = max(1, N - window_size + 1)

            self.token_cache.append(tokens)
            self.pos_cache.append(positions_raw)
            self.meta.append((N, max_start))
            self.label_cache.append(rot_class)

    def __len__(self):
        return sum(max_start for _, max_start in self.meta)

    def __getitem__(self, idx: int) -> dict:
        # ---------------------------------------------------
        # 1) Map global idx → (Index im Cache i, start innerhalb dieses Modells)
        # ---------------------------------------------------
        cum = 0
        for i, (N, max_start) in enumerate(self.meta):
            if idx < cum + max_start:
                start = idx - cum
                break
            cum += max_start

        tokens   = self.token_cache[i]    # [N_i, 2]
        pos_raw  = self.pos_cache[i]      # [N_i, 3]
        rot_class = self.label_cache[i]   # Integer 0…5, für dieses gesamte Modell

        # ---------------------------------------------------
        # 2) Fenster herausschneiden
        # ---------------------------------------------------
        window_tokens    = tokens[start : start + self.window_size]       # [W', 2]
        window_positions = pos_raw [start : start + self.window_size]      # [W', 3]
        W = window_tokens.shape[0]

        # Padding, falls W' < window_size
        if W < self.window_size:
            pad_len = self.window_size - W
            pad_tokens = torch.zeros(pad_len, window_tokens.size(1), dtype=window_tokens.dtype)
            window_tokens = torch.cat([window_tokens, pad_tokens], dim=0)       # [window_size,2]
            pad_pos    = torch.zeros(pad_len, window_positions.size(1), dtype=window_positions.dtype)
            window_positions = torch.cat([window_positions, pad_pos], dim=0)   # [window_size,3]
            W = self.window_size

        # ---------------------------------------------------
        # 3) Augment (optional)
        # ---------------------------------------------------
        if self.augment:
            noise = torch.randn_like(window_tokens) * 0.01
            window_tokens = window_tokens + noise

        # ---------------------------------------------------
        # 4) Positionsdaten normalisieren
        # ---------------------------------------------------
        abs_idx    = window_positions[:, 0]        # [W], Rohwerte
        levels     = window_positions[:, 1].long() # [W], Integer 0..15
        pos_in_lvl = window_positions[:, 2]        # [W]

        # 4.1) abs_norm = abs_idx / total_tokens → [W,1]
        abs_norm = (abs_idx / float(self.total_tokens)).unsqueeze(-1)

        # 4.2) p_norm = pos_in_lvl / level_sizes[level] → [W,1]
        p_norm = torch.zeros_like(pos_in_lvl)
        for lvl, size in self.level_sizes.items():
            mask = (levels == lvl)
            if mask.any():
                p_norm[mask] = pos_in_lvl[mask] / float(size)
        p_norm = p_norm.unsqueeze(-1)

        # ---------------------------------------------------
        # 5) Gib alle vier Features plus Rotation‐Label zurück
        # ---------------------------------------------------
        return {
            "tokens":    window_tokens,             # FloatTensor [W, 2]
            "abs_norm":  abs_norm,                  # FloatTensor [W, 1]
            "p_norm":    p_norm,                    # FloatTensor [W, 1]
            "levels":    levels,                    # LongTensor  [W]
            "label_rot": torch.tensor(rot_class)    # LongTensor  []  (einzelner Integer 0…5)
        }
