import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

class MaskedAutoencoderViT(nn.Module):
    """
    Spatio-Temporal Masked Autoencoder based on the Prithvi architecture.
    Utilizes 3D Tubelet Embeddings for multispectral time-series remote sensing data.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=6,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, num_frames=3, tubelet_size=1, **kwargs):
        super().__init__()

        # --- Encoder: Spatio-Temporal Tubelet Embedding ---
        # Implementation of 3D Convolutional projection for space-time volumes
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(tubelet_size, patch_size, patch_size), 
            stride=(tubelet_size, patch_size, patch_size)
        )
        
        num_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --- Decoder: Latent Projection to Pixel Space ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Output head: Projects to flattened patch dimensions (C * T * H * W)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            tubelet_size * patch_size * patch_size * in_chans, 
            bias=True
        )

    def forward(self, x):
        """
        Forward pass for inference.
        Input x shape: [Batch, Channels, Time, Height, Width]
        """
        # Patch and Tubelet Embedding
        x = self.patch_embed(x)  # [B, E, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, L, E]
        
        # Token augmentation and positional encoding
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        # Latent space encoding
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # Decoding and reconstruction
        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        # Return signature consistent with MAE training (loss, pred, mask)
        return None, x, None