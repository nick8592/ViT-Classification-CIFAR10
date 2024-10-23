import torch
import torch.nn as nn

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# N -> Number of Patches = IH/P * IW/P
# S -> Sequence Length   = IH/P * IW/P + 1 or N + 1 (extra 1 is of Classification Token)
# Q -> Query Sequence length (equal to S for self-attention)
# K -> Key Sequence length   (equal to S for self-attention)
# V -> Value Sequence length (equal to S for self-attention)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H
# CL -> Number of Classes

#TODO Patch Embedding
class EmbedLayer(nn.Module):
    """
    Embedding an Image.
    Conv2D breaks image into patches.
    Add positional embedding vector into pathches embeddings.
    Add classification token
    """
    def __init__(self, n_channels: int, embed_dim: int, image_size: int, 
                 patch_size: int, dropout: float=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size) # patch encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True) # learnable positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True) # classification token
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x) # (B, C, IH, IW) -> (B, E, IH/P, IW/P) split image into patches
        x = x.reshape([B, x.shape[1], -1]) # (B, E, IH/P, IW/P) -> (B, E, IH/P*IW/P) -> (B, E, N) flatten the patches
        x = x.permute(0, 2, 1) # (B, E, N) -> (B, N, E) # rearrange to put sequence dimension in the middle
        x = x + self.pos_embedding # (B, N, E) -> (B, N, E) add positional embedding
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1) # (B, N, E) -> (B, N+1, E) -> (B, S, E) add classification token at the start of every sequence
        x = self.dropout(x)
        return x
#TODO Positional Encoding
#TODO Transformer Encoder
#TODO Classification Head

def test(model: nn.Module, n_channels: int, embed_dim: int, image_size: int, patch_size: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.rand((1, n_channels, image_size, image_size)).to(device)
    ViT = model(n_channels, embed_dim, image_size, patch_size).to(device)
    output = ViT(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test(EmbedLayer, 3, 32, 32, 4)