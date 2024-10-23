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

class EmbedLayer(nn.Module):
    """
    Class for Embedding an Image.
    It breaks image into patches and embeds patches using a Conv2D Operation (Works same as the Linear layer).
    Next, a learnable positional embedding vector is added to all the patch embeddings to provide spatial position.
    Finally, a classification token is added which is used to classify the image.

    Parameters:
        n_channels (int) : Number of channels of the input image
        embed_dim  (int) : Embedding dimension
        image_size (int) : Image size
        patch_size (int) : Patch size
        dropout  (float) : dropout value

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH
    
    Returns:
        Tensor: Embedding of the image of shape B, S, E
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
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_attention_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads) # Quaries projection (learnable weight)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads) # Keys projection (learnable weight)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads) # Values projection (learnable weight)
        self.out_projection     = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim) # Out projection (learnable weight)

    def forward(self, x):
        b, s, e = x.shape # in case fo self-attention Q, K, V are all equal to S

        # Linear projection (learnable weight)
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim) # (B, Q, E) -> (B, Q, (H*HE)) -> (B, Q, H, HE)
        xq = xq.permute(0, 2, 1, 3) # (B, Q, H, HE) -> (B, H, Q, HE)
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim) # (B, K, E) -> (B, K, (H*HE)) -> (B, K, H, HE)
        xk = xk.permute(0, 2, 1, 3) # (B, K, H, HE) -> (B, H, K, HE)
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim) # (B, V, E) -> (B, V, (H*HE)) -> (B, V, H, HE)
        xv = xv.permute(0, 2, 1, 3) # (B, V, H, HE) -> (B, H, V, HE)

        # Computer Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2) # (B, H, K, HE) -> (B, H, HE, K) K^T
        x_attention = torch.matmul(xq, xk) # (B, H, Q, HE) * (B, H, HE, K) -> (B, H, Q, K) dot product (Q*K^T)

        x_attention /= float(self.head_embed_dim) ** 0.5 # scale presoftmax values for stability (Q*K^T/dk^0.5)

        x_attention = torch.softmax(x_attention, dim=-1) # compute attention matrix (softmax(QK^T/dk^0.5))

        x = torch.matmul(x_attention, xv) # (B, H, Q, K) * (B, H, V, HE) -> (B, H, Q, HE) dot product (softmax(QK^T/dk^0.5)V)

        # # Format the output
        x = x.permute(0, 2, 1, 3) # (B, H, Q, HE) -> (B, Q, H, HE)
        x = x.reshape(b, s, e) # (B, Q, H, HE) -> (B, Q, (H*HE))

        x = self.out_projection(x) # (B, Q, (H*HE)) -> (B, Q, E)
        return x

def test(n_channels: int, embed_dim: int, image_size: int, 
         patch_size: int, n_attention_heads: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.rand((1, n_channels, image_size, image_size)).to(device)
    embed = EmbedLayer(n_channels, embed_dim, image_size, patch_size).to(device)
    atten = SelfAttention(embed_dim, n_attention_heads).to(device)

    patches = embed(x)
    atten_weight = atten(patches)

    print(f"Input shape: {x.shape}")
    print(f"Patches shape: {patches.shape}")
    print(f"Attention shape: {atten_weight.shape}")

if __name__ == "__main__":
    test(n_channels=3, embed_dim=32, image_size=32, 
         patch_size=4, n_attention_heads=4)