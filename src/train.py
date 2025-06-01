import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1) Imports: Dataset & Model definitions
from dataloader import SimpleTokenDataset
from transformer import SANETokenAutoencoder

# -------------------------------------------------------------------------
# GPU‐Stability Settings
# -------------------------------------------------------------------------
os.environ["PYTORCH_ENABLE_FLASH_ATTN"] = "0"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


if __name__ == '__main__':
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    token_dir    = "../prepared_objects_first_5_levels"
    model_ids    = ["asteroid6__base_000_000_000__checkpoints__final"] # thats what you train on
    window_size  = 256
    batch_size   = 8
    lr           = 1e-4
    epochs       = 5
    log_interval = 50

    # -----------------------------
    # Ensure "checkpoints" folder exists
    # -----------------------------
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    dataset = SimpleTokenDataset(
        token_dir=token_dir,
        model_ids=model_ids,
        window_size=window_size,
        augment=True  # Set to True if you want to add noise or jitter to the tokens
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------
    # Device, Model, Optimizer
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SANETokenAutoencoder(
        token_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        level_embed_dim=16
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        loop = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}', unit='Batch')
        for batch_idx, batch in enumerate(loop, 1):
            # ---------------------------------------------------
            # 1) Extract batch tensors and move them to device
            #    - tokens:   [B, W, 2]
            #    - abs_norm: [B, W, 1]
            #    - p_norm:   [B, W, 1]
            #    - levels:   [B, W]
            # ---------------------------------------------------
            tokens   = batch['tokens'].to(device)
            abs_norm = batch['abs_norm'].to(device)
            p_norm   = batch['p_norm'].to(device)
            levels   = batch['levels'].to(device)

            # ---------------------------------------------------
            # 2) Sanity‐Checks: ensure no NaN or Inf in inputs
            # ---------------------------------------------------
            if torch.isnan(tokens).any() or torch.isinf(tokens).any():
                raise ValueError(f"NaN/Inf detected in tokens at batch {batch_idx}")
            if torch.isnan(abs_norm).any() or torch.isinf(abs_norm).any():
                raise ValueError(f"NaN/Inf detected in abs_norm at batch {batch_idx}")
            if torch.isnan(p_norm).any() or torch.isinf(p_norm).any():
                raise ValueError(f"NaN/Inf detected in p_norm at batch {batch_idx}")

            optimizer.zero_grad()

            # ---------------------------------------------------
            # 3) Forward‐Pass & Loss Computation
            # ---------------------------------------------------
            recon = model(tokens, abs_norm, p_norm, levels)      # [B, W, 2]
            loss  = F.mse_loss(recon, tokens)                    # MSE loss on 2D tokens
            loss.backward()

            # ---------------------------------------------------
            # 4) Gradient‐Clipping & Optimizer Step
            # ---------------------------------------------------
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % log_interval == 0:
                avg_loss = epoch_loss / batch_idx
                loop.set_postfix(
                    batch_loss=f"{loss.item():.6f}",
                    avg_loss=f"{avg_loss:.6f}"
                )

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} completed in {elapsed:.1f}s – Avg Loss: {avg_loss:.6f}")

        # ---------------------------------------------------
        # 5) Save Checkpoint to "checkpoints" folder
        # ---------------------------------------------------
        checkpoint_name = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch:02d}.pt")
        torch.save(model.state_dict(), checkpoint_name)
        # Optionally, log that the checkpoint has been saved
        # e.g., print(f"Saved checkpoint: {checkpoint_name}")
