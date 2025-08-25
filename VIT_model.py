import torch
import torch.nn as nn

# 1. Create a class that inherits from nn.Module


class PatchEmbeddings(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        img_size (int): Image size.
        in_channels (int): Number of input channels, e.g. 3 for RGB.
        patch_size (int): Size of patches to convert image into.
        embedding_dim (int): Size of embedding to turn image into.
    """
    # 2. Initialize the class with appropriate parameters

    def __init__(self, img_size: int = 224, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
        super().__init__()

        # Store patch_size as instance attribute
        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        # NOTE: In ViT paper, they use a convolutional layer to create patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # Only flatten the feature map dimensions not number of patches
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct size
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x = self.patcher(x)
        x = self.flatten(x)
        # 6. Permute the flattened patches to be batch_size, number_of_patches, embedding_dim
        # adjust so the embedding is on the final dimension [batch_size, P^2, embedding_dim]
        return x.permute(0, 2, 1)

# 1. Create a class that inherits from nn.Module


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 2. Initialize the class with appropriate parameters

    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, mlp_size: int = 3072, attn_dropout: float = 0, mlp_dropout: float = 0.1):
        super().__init__()

        # 3. Create MHSA layer (self-attention)
        self.msa_block = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads,
                                               dropout=attn_dropout,
                                               batch_first=True)  # Does our batch size come first?

        # 4. Create MLP layer (feed-forward)
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=mlp_dropout)
        )

    # 5. Create a forward() method
    def forward(self, x):
        # Create residual connection(s)
        x = self.msa_block(query=x,  # Query, key and value are all the same for self-attention
                           key=x,
                           value=x,
                           need_weights=False)[0] + x  # Add skip connection (Residual connection)
        x = self.mlp_block(x) + x  # Add skip connection (Residual connection)

        return x


class ViT(nn.Module):
    """Creates a Vision Transformer model."""

    def __init__(self, img_size: int = 224, in_channels: int = 3, patch_size: int = 16, num_transformer_layers: int = 12,  # Based on ViT-Base
                 embedding_dim: int = 768, mlp_size: int = 3072, num_heads: int = 12, attn_dropout: float = 0,
                 mlp_dropout: float = 0.1, embedding_dropout: float = 0.1, num_classes: int = 1000):  # Default to ImageNet
        super().__init__()

        # 1. Create patch embedding layer
        self.patch_embedding = PatchEmbeddings(img_size=img_size,
                                               in_channels=in_channels,
                                               patch_size=patch_size,
                                               embedding_dim=embedding_dim)

        # 2. Create class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)

        # 3. Create positional embedding
        # NOTE: The paper doesn't specify where to get the positional embeddings from,
        # but PyTorch's implementation uses nn.Parameter
        num_patches = (img_size * img_size) // patch_size**2  # N = HW/P^2
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim),  # Add 1 for class token
                                               requires_grad=True)

        # 4. Create embedding dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 5. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           attn_dropout=attn_dropout,
                                                                           mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # 6. Create Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    # 7. Create a forward() method
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]

        # Create class token and expand it to the batch size
        class_token = self.class_token.expand(
            batch_size, -1, -1)  # -1 will infer the size

        # Perform patch embedding
        x = self.patch_embedding(x)

        # Concatenate class token and patch embeddings
        x = torch.cat((class_token, x), dim=1)

        # Add positional embeddings
        x = self.position_embedding + x

        # Apply embedding dropout
        x = self.embedding_dropout(x)

        # Pass through Transformer Encoder blocks
        x = self.transformer_encoder(x)

        # Pass through Classifier head
        # We only need the class token for classification
        x = self.classifier(x[:, 0])  # Get the first token of the output

        return x
