import torch
import torch.nn as nn
import torchvision.models as models

class NeuroTeX(nn.Module):
    """
    NeuroTeX: A vision-to-LaTeX transformer model that uses a CNN (ResNet) to encode
    input images and a Transformer decoder to autoregressively generate LaTeX tokens.
    """
    def __init__(self, vocab_size, max_len=128, embed_dim=512, nhead=8, num_layers=4, encoder_name="resnet18"):
        super(NeuroTeX, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        # Vision Encoder
        # Load pretrained ResNet and remove last two layers (avgpool & fully connected)
        if encoder_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.cnn_encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool & FC
            input_channels = 512
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        # Project CNN output to the embedding dimension expected by the transformer
        self.encoder_conv = nn.Conv2d(input_channels, embed_dim, kernel_size=1)  # [B, E, H, W]

        # Token embedding table for LaTeX vocabulary
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # Learnable positional embeddings for up to max_len tokens
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))

        # Transformer Decoder
        # Uses PyTorch's built-in Transformer decoder layer with multi-head attention
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output Projection
        # Final layer to project transformer outputs to vocab size logits
        self.output_fc = nn.Linear(embed_dim, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        """
        Generates an upper-triangular matrix filled with -inf used for causal masking.
        This prevents the decoder from attending to future tokens.
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, image, tgt_seq):
        """
        Forward pass for training or inference.

        Args:
            image: Tensor of shape [B, 3, H, W] — input images
            tgt_seq: Tensor of shape [B, T] — target token IDs

        Returns:
            Tensor of shape [B, T, vocab_size] — unnormalized logits for each token
        """
        B, T = tgt_seq.shape

        # Encode the input image
        x = self.cnn_encoder(image)              # [B, 512, H/32, W/32]
        x = self.encoder_conv(x)                 # [B, E, H', W']
        x = x.flatten(2).permute(2, 0, 1)        # Flatten spatial dims -> [S, B, E], where S = H'*W'

        # Embed target tokens
        tgt = self.token_embed(tgt_seq)          # [B, T, E]
        tgt = tgt + self.pos_embed[:, :T, :]     # Add positional encoding
        tgt = tgt.permute(1, 0, 2)               # [T, B, E] for transformer

        # Create a causal mask to prevent looking ahead
        tgt_mask = self._generate_square_subsequent_mask(T).to(tgt.device)  # [T, T]

        # Decode using Transformer
        decoded = self.transformer_decoder(tgt, x, tgt_mask=tgt_mask)       # [T, B, E]
        logits = self.output_fc(decoded)                                    # [T, B, V]
        return logits.permute(1, 0, 2)                                      # [B, T, V] for cross-entropy
