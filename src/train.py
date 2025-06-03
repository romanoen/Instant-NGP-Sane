import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import SimpleTokenDataset       # Passen: Dateiname, falls anders
from transformer import SANETokenAutoencoderWithRotation  # Passen

# ----------------------------------------
# 1) Hyperparameter & Pfade
# ----------------------------------------
token_dir    = "../prepared_objects_first_4_levels"
model_ids    = ["asteroid6__base_000_000_000__checkpoints__final",
                "asteroid6__compound_090_000_090__checkpoints__final",
                "asteroid6__x_180_000_000__checkpoints__final",
                "asteroid6__y_000_180_000__checkpoints__final",
                "asteroid6__z_000_000_120__checkpoints__final",
                "asteroid6__z_000_000_240__checkpoints__final"]  # 6 Rotationen
window_size  = 256
batch_size   = 8
lr           = 1e-4
epochs       = 10
lambda_rot   = 0.5  # Gewicht des Rotations‐Loss gegenüber Rekonstruktions‐Loss
checkpoint_dir = "../checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
# 2) Dataset & DataLoader
# ----------------------------------------
dataset = SimpleTokenDataset(
    token_dir=token_dir,
    model_ids=model_ids,
    window_size=window_size,
    augment=False  # Bei Bedarf auf True setzen (Rauschen)
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print(f"Dataset size (windows): {len(dataset)}, num_batches: {len(dataloader)}")

# ----------------------------------------
# 3) Modell & Optimizer initialisieren
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

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ----------------------------------------
# 4) Training Loop
# ----------------------------------------
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for batch_idx, batch in enumerate(loop, 1):
        tokens   = batch["tokens"].to(device)     # [B, W, 2]
        abs_norm = batch["abs_norm"].to(device)   # [B, W, 1]
        p_norm   = batch["p_norm"].to(device)     # [B, W, 1]
        levels   = batch["levels"].to(device)     # [B, W]
        labels   = batch["label_rot"].to(device)  # [B] Integer in [0..5]

        optimizer.zero_grad()

        # Forward‐Pass
        recon, logits_rot = model(tokens, abs_norm, p_norm, levels)
        # 1) Rekonstruktions‐Loss (MSE über alle B×W×2 Werte)
        loss_recon = F.mse_loss(recon, tokens)
        # 2) Rotations‐Loss (CrossEntropy über B × 6‐Logits)
        loss_rot   = F.cross_entropy(logits_rot, labels)
        # 3) Gesamt‐Loss
        loss = loss_recon + lambda_rot * loss_rot

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        if batch_idx % 50 == 0:
            avg = epoch_loss / batch_idx
            loop.set_postfix(batch_loss=f"{loss.item():.6f}", avg_loss=f"{avg:.6f}")

    elapsed = time.time() - start_time
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch}/{epochs} completed in {elapsed:.1f}s – Avg Loss: {avg_loss:.6f}")

os.makedirs(checkpoint_dir, exist_ok=True)
save_path = os.path.join(checkpoint_dir, "sane_asteroid6_with_rotation_final.pt")

torch.save(model.state_dict(), save_path)
print(f"Modell gespeichert in: {save_path}")