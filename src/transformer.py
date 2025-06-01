import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block with:
      1) Multi‐head self‐attention (with residual + LayerNorm)
      2) Position‐wise feedforward network (with residual + LayerNorm)

    Input & output shape: [B, W, D], where
      B = batch size,
      W = sequence length (window),
      D = model dimension (d_model).
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()

        # 1) Multi‐head self‐attention layer.
        #    - embed_dim = d_model: dimension of each token embedding
        #    - num_heads = nhead: how many attention heads to use
        #    - dropout = dropout: dropout probability inside attention
        #    - batch_first = True → expect input shape [B, W, D].
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 2) LayerNorm applied after adding residual from attention.
        #    Normalizes across the last dimension (d_model).
        self.ln1 = nn.LayerNorm(d_model)

        # 3) Position‐wise feedforward network:
        #    - Linear layer projects from d_model → dim_feedforward
        #    - GELU nonlinearity
        #    - Dropout for regularization
        #    - Linear layer projects back from dim_feedforward → d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # 4) LayerNorm applied after adding residual from feedforward.
        self.ln2     = nn.LayerNorm(d_model)

        # 5) Dropout layer used for both attention output and feedforward output.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, W, D] input tensor of token embeddings.

        Returns:
          x: [B, W, D] output tensor after one Encoder block.
        """
        # -----------------------------
        # 1) Self‐Attention sublayer
        # -----------------------------
        # `self.attn(x, x, x)` performs queries, keys, and values all = x (self‐attention).
        # attn_out: [B, W, D], _ = attention weights (ignored here).
        attn_out, _ = self.attn(x, x, x)            

        # Add & Norm: residual connection + LayerNorm
        #   - Residual: x + dropout(attn_out)
        #   - Then apply LayerNorm over dimension D
        # Resulting shape: [B, W, D]
        x = self.ln1(x + self.dropout(attn_out))    

        # -----------------------------
        # 2) Feedforward sublayer
        # -----------------------------
        # Apply position‐wise feedforward network to the normalized tensor
        # ff_out: [B, W, D]
        ff_out = self.ff(x)                         

        # Add & Norm: residual connection + LayerNorm
        #   - Residual: x + dropout(ff_out)
        #   - Then apply LayerNorm over dimension D
        # Final shape: [B, W, D]
        x = self.ln2(x + self.dropout(ff_out))      

        return x


class SANETokenAutoencoder(nn.Module):
    """
    A token‐level autoencoder that uses a Transformer encoder to reconstruct 2D tokens.
    Incorporates learned positional embeddings based on:
      - absolute normalized position (abs_norm)
      - periodic normalized position (p_norm)
      - discrete INGP "level" embeddings (levels 0..4)

    Flow:
      1) Project input tokens [B, W, token_dim=2] → dense embeddings [B, W, d_model].
      2) Build a positional embedding from (abs_norm, p_norm, levels) → [B, W, d_model].
      3) Sum token‐embedding + positional‐embedding → [B, W, d_model].
      4) Pass through a stack of TransformerBlocks.
      5) Decode back to token_dim = 2 via a small MLP.
    """
    def __init__(
        self,
        token_dim: int = 2,          # Dimensionality of each token (e.g., 2 for (x,y) coordinates)
        d_model: int   = 64,         # Transformer model dimension
        nhead: int     = 4,          # Number of attention heads
        num_layers: int= 2,          # Number of TransformerBlocks in the encoder stack
        dim_feedforward: int = 256,  # Hidden dimension in the feedforward sublayer
        dropout: float = 0.1,        # Dropout probability in both attention & feedforward
        level_embed_dim: int = 16,   # Embedding dimension for discrete "level" input
    ):
        super().__init__()

        # ---------------------------------------------------
        # 1) Token‐Embedding: maps each 2‐D token → d_model‐dim vector
        #    Input:  tokens [B, W, token_dim=2]
        #    Output: tok_emb [B, W, d_model=64]
        # ---------------------------------------------------
        self.token_embed = nn.Linear(token_dim, d_model)

        # ---------------------------------------------------
        # 2) Learned embedding for "levels" (integer values 0..4).
        #    Input:  levels [B, W] (LongTensor, values in {0,1,2,3,4})
        #    Output: lvl_emb [B, W, level_embed_dim=16]
        # ---------------------------------------------------
        self.level_emb = nn.Embedding(num_embeddings=5, embedding_dim=level_embed_dim)

        # ---------------------------------------------------
        # 3) Linear projection for combined positional info:
        #      - abs_norm: [B, W, 1]
        #      - p_norm:   [B, W, 1]
        #      - lvl_emb:  [B, W, level_embed_dim=16]
        #    Concatenate along last dimension → [B, W, 1 + 1 + 16 = 18]
        #    Then project to d_model (64).
        #    Input dim = 18, output dim = 64
        # ---------------------------------------------------
        self.pos_proj = nn.Linear(1 + 1 + level_embed_dim, d_model)

        # ---------------------------------------------------
        # 4) Transformer‐Encoder: stack of `num_layers` × TransformerBlock
        # ---------------------------------------------------
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # ---------------------------------------------------
        # 5) Decoder: simple 2‐layer MLP
        #    Input:  [B, W, d_model=64]
        #    Hidden: dim_feedforward=256, activation GELU
        #    Output: [B, W, token_dim=2]
        # ---------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, token_dim)
        )

    def forward(
        self,
        tokens: torch.Tensor,   # [B, W, token_dim=2]
        abs_norm: torch.Tensor, # [B, W, 1]  (e.g., absolute position normalized)
        p_norm: torch.Tensor,   # [B, W, 1]  (e.g., periodic position normalized)
        levels: torch.Tensor     # [B, W]     (int64 values in 0..4)
    ) -> torch.Tensor:
        """
        Perform a forward pass through the autoencoder.

        Arguments:
          tokens:   FloatTensor of shape [B, W, 2]  — input tokens to reconstruct.
          abs_norm: FloatTensor of shape [B, W, 1]  — absolute normalized positions.
          p_norm:   FloatTensor of shape [B, W, 1]  — periodic normalized positions.
          levels:   LongTensor of shape [B, W]     — learned discrete "level" indices.

        Returns:
          recon: FloatTensor [B, W, 2] = reconstructed tokens.
        """
        # Capture batch size B and sequence length W.
        B, W, _ = tokens.shape

        # -----------------------------
        # 1) Token‐Embedding
        # -----------------------------
        # Project each 2‐D token into a d_model‐dim embedding.
        # tok_emb: [B, W, 64]
        tok_emb = self.token_embed(tokens)

        # -----------------------------
        # 2) Positional Embedding
        #    2.1) Embed discrete levels → [B, W, 16]
        #    2.2) Concatenate abs_norm/p_norm (each [B, W, 1]) with lvl_emb [B, W, 16]
        #         → pos_cat: [B, W, 18]
        #    2.3) Linear proj from 18 → 64 → pos_emb: [B, W, 64]
        # -----------------------------
        # 2.1) Level embeddings
        lvl_emb = self.level_emb(levels)  # [B, W, 16]

        # 2.2) Concatenate along the last dimension: [abs_norm, p_norm, lvl_emb]
        # pos_cat: [B, W, 1 + 1 + 16 = 18]
        pos_cat = torch.cat([abs_norm, p_norm, lvl_emb], dim=-1)  # [B, W, 18]

        # 2.3) Project to d_model dimension
        # pos_emb: [B, W, 64]
        pos_emb = self.pos_proj(pos_cat)

        # -----------------------------
        # 3) Sum Token + Positional embeddings
        #    Combined embedding x: [B, W, 64]
        # -----------------------------
        x = tok_emb + pos_emb  # element‐wise sum [B, W, 64]

        # -----------------------------
        # 4) Pass through each TransformerBlock
        #    Each block preserves shape [B, W, 64].
        # -----------------------------
        for blk in self.encoder_blocks:
            x = blk(x)

        # -----------------------------
        # 5) Decode back to 2‐D tokens
        #    MLP: [B, W, 64] → [B, W, 256] → GELU → [B, W, 2]
        # -----------------------------
        recon = self.decoder(x)  # [B, W, 2]

        return recon
