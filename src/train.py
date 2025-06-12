import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

from dataloader import SimpleTokenDataset       # Adjust: filename if different
from transformer import SANETokenAutoencoderWithRotation  # Adjust if needed
from utils import get_filenames

# ----------------------------------------
# 1) Hyperparameters & Paths
# ----------------------------------------
token_dir      = "../prepared_objects_first_4_levels"
model_ids      = get_filenames(root_folder=token_dir)
window_size    = 256
batch_size     = 8
learning_rate  = 1e-4
epochs         = 10
lambda_rot     = 0.5   # Weight of rotation loss relative to reconstruction loss
val_split      = 0.2   # Fraction for validation set
checkpoint_dir = "../checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
# 2) Dataset & DataLoaders (with split)
# ----------------------------------------
dataset = SimpleTokenDataset(
    token_dir=token_dir,
    model_ids=model_ids,
    window_size=window_size,
    augment=True  # Apply noise only during training
)

# Split into training and validation sets
total_size = len(dataset)
val_size = int(val_split * total_size)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ----------------------------------------
# 3) Initialize Model, Optimizer & AMP Scaler
# ----------------------------------------
model = SANETokenAutoencoderWithRotation(
    token_dim=2,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    level_embed_dim=16,
    num_rot_classes=6,
    rot_hidden=32
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Use new torch.amp API for device-specific autocast and GradScaler
dtype = 'cuda' if device.type == 'cuda' else 'cpu'
scaler = GradScaler(device=dtype)
best_val_loss = float('inf')

os.makedirs(checkpoint_dir, exist_ok=True)

# ----------------------------------------
# 4) Training and Validation Loop with AMP
# ----------------------------------------
for epoch in range(1, epochs + 1):
    # --- Training ---
    model.train()
    train_loss = 0.0
    start_time = time.time()

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
    for batch in train_bar:
        tokens   = batch["tokens"].to(device)
        abs_norm = batch["abs_norm"].to(device)
        p_norm   = batch["p_norm"].to(device)
        levels   = batch["levels"].to(device)
        labels   = batch["label_rot"].to(device)

        optimizer.zero_grad()

        # Forward with automatic mixed precision
        with autocast(device_type=dtype):
            recon, logits_rot = model(tokens, abs_norm, p_norm, levels)
            loss_recon = F.mse_loss(recon, tokens)
            loss_rot   = F.cross_entropy(logits_rot, labels, label_smoothing=0.1)
            loss = loss_recon + lambda_rot * loss_rot

        # Scaled backward
        scaler.scale(loss).backward()

        # Unscale gradients and clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_time = time.time() - start_time

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            tokens   = batch["tokens"].to(device)
            abs_norm = batch["abs_norm"].to(device)
            p_norm   = batch["p_norm"].to(device)
            levels   = batch["levels"].to(device)
            labels   = batch["label_rot"].to(device)

            recon, logits_rot = model(tokens, abs_norm, p_norm, levels)
            loss_recon = F.mse_loss(recon, tokens)
            loss_rot   = F.cross_entropy(logits_rot, labels)
            loss = loss_recon + lambda_rot * loss_rot
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (Time: {train_time:.1f}s)")

    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  ↳ New best! Model saved: {save_path}")

# Final save of the last model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pt"))
print("Training complete. Final and best models have been saved.")
