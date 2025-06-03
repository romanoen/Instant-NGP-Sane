import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ln2     = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)                 # [B, W, D]
        x = self.ln1(x + self.dropout(attn_out))         # [B, W, D]
        ff_out = self.ff(x)                              # [B, W, D]
        x = self.ln2(x + self.dropout(ff_out))           # [B, W, D]
        return x

class SANETokenAutoencoderWithRotation(nn.Module):
    def __init__(
        self,
        token_dim: int = 2,
        d_model: int   = 64,
        nhead: int     = 4,
        num_layers: int= 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        level_embed_dim: int = 16,
        num_rot_classes: int = 6,
        rot_hidden: int = 32,
    ):
        super().__init__()

        # -------------------------
        # 1) Token‐Embedding
        # -------------------------
        self.token_embed = nn.Linear(token_dim, d_model)

        # -------------------------
        # 2) Level‐Embedding (0..4)
        # -------------------------
        self.level_emb = nn.Embedding(num_embeddings=5, embedding_dim=level_embed_dim)

        # -------------------------
        # 3) Positional‐Proj: (1 + 1 + level_embed_dim) → d_model
        # -------------------------
        self.pos_proj = nn.Linear(1 + 1 + level_embed_dim, d_model)

        # -------------------------
        # 4) Transformer‐Encoder (num_layers × TransformerBlock)
        # -------------------------
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout)
            for _ in range(num_layers)
        ])

        # -------------------------
        # 5) Decoder: [B, W, d_model] → [B, W, token_dim]
        # -------------------------
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, token_dim)
        )

        # -------------------------
        # 6) Rotation‐Head: [B, d_model] → rot_hidden → num_rot_classes
        # -------------------------
        self.rotation_head = nn.Sequential(
            nn.Linear(d_model, rot_hidden),
            nn.ReLU(),
            nn.Linear(rot_hidden, num_rot_classes)
        )

    def forward(
        self,
        tokens: torch.Tensor,    # [B, W, 2]
        abs_norm: torch.Tensor,  # [B, W, 1]
        p_norm: torch.Tensor,    # [B, W, 1]
        levels: torch.Tensor     # [B, W], LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward‐Pass liefert:
          1) recon:    [B, W, 2] (rekonstruierte Tokens)
          2) logits_rot: [B, num_rot_classes] (Rotation‐Logits für jede Klasse)
        """
        B, W, _ = tokens.shape

        # -------------------------
        # a) Token‐Embedding
        # -------------------------
        tok_emb = self.token_embed(tokens)  # [B, W, d_model]

        # -------------------------
        # b) Positional‐Embedding
        # -------------------------
        lvl_emb = self.level_emb(levels)    # [B, W, level_embed_dim]
        pos_cat = torch.cat([abs_norm, p_norm, lvl_emb], dim=-1)  # [B, W, 1+1+lev_emb_dim]
        pos_emb = self.pos_proj(pos_cat)    # [B, W, d_model]

        # -------------------------
        # c) Sum Token + Position
        # -------------------------
        x = tok_emb + pos_emb  # [B, W, d_model]

        # -------------------------
        # d) Transformer‐Encoder
        # -------------------------
        for blk in self.encoder_blocks:
            x = blk(x)        # [B, W, d_model]

        # -------------------------
        # e) Rekonstruktions‐Decoder
        # -------------------------
        recon = self.decoder(x)  # [B, W, token_dim=2]

        # -------------------------
        # f) Rotation‐Head (Mean‐Pooling → MLP)
        # -------------------------
        z = x.mean(dim=1)         # [B, d_model]
        logits_rot = self.rotation_head(z)  # [B, num_rot_classes=6]

        return recon, logits_rot
