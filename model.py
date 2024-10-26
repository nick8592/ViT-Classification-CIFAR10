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
    """
    Class for computing self attention Self-Attention

    Parameters:
        embed_dim (int)        : Embedding dimension
        n_attention_heads (int): Number of attention heads to use for performing MultiHeadAttention
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output after Self-Attention Module of shape B, S, E
    """ 
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
        return x, x_attention
    
class Encoder(nn.Module):
    """
    Class for creating an encoder layer

    Parameters:
        embed_dim (int)         : Embedding dimension
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout (float)         : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output of the encoder block of shape B, S, E
    """
    def __init__(self, embed_dim: int, n_attention_heads: int, 
                 forward_mul: float, dropout: float=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x, att_mat = self.attention(self.norm1(x))
        x = x + self.dropout1(x) # skip connection
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x))))) # skip connection
        return x, att_mat
    
class Classifier(nn.Module):
    """
    Classification module of the Vision Transformer. Uses the embedding of the classification token to generate logits.

    Parameters:
        embed_dim (int) : Embedding dimension
        n_classes (int) : Number of classes
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Logits of shape B, CL
    """ 
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        # new architectures skip fc1 and activations and directly apply fc2
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x):
        x = x[:, 0, :] # (B, S, E) -> (B, E) get CLS token
        x = self.fc1(x) # (B, E) -> (B, E)
        x = self.activation(x) # (B, E) -> (B, E)
        x = self.fc2(x) # (B, E) -> (B, CL)
        return x
    
class VisionTransformer(nn.Module):
    """
    Vision Transformer Class.

    Parameters:
        n_channels (int)        : Number of channels of the input image
        embed_dim  (int)        : Embedding dimension
        n_layers   (int)        : Number of encoder blocks to use
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        image_size (int)        : Image size
        patch_size (int)        : Patch size
        n_classes (int)         : Number of classes
        dropout  (float)        : dropout value
    
    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Logits of shape B, CL
    """ 
    def __init__(self, n_channels: int, embed_dim: int, n_layers: int, 
                 n_attention_heads: int, forward_mul: float, image_size: int, 
                 patch_size: int, n_classes: int, dropout: float=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim) # final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)

        self.apply(vit_init_weights)

    def forward(self, x):
        x = self.embedding(x)
        for i, block in enumerate(self.encoder):
            x, att_mat = block(x)
            att_mat_full = att_mat if i == 0 else torch.cat((att_mat_full, att_mat), dim=0)
        x = self.norm(x)
        x = self.classifier(x)
        return x, att_mat_full
    
def vit_init_weights(m): 
    """
    function for initializing the weights of the Vision Transformer.
    """    

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)

def test(n_channels: int, embed_dim: int, n_layers: int, 
         n_attention_heads: int, forward_mul: float, image_size: int, 
         patch_size: int, n_classes: int, dropout: float=0.1):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.rand((1, n_channels, image_size, image_size)).to(device)
    embed = EmbedLayer(n_channels, embed_dim, image_size, patch_size).to(device)
    # atten = SelfAttention(embed_dim, n_attention_heads).to(device)
    encoder = Encoder(embed_dim, n_attention_heads, forward_mul).to(device)
    classifier = Classifier(embed_dim, n_classes).to(device)
    vit = VisionTransformer(n_channels, embed_dim, n_layers, n_attention_heads,
                            forward_mul, image_size, patch_size, n_classes, dropout).to(device)

    patches = embed(x)
    # atten_weight = atten(patches)
    enc_out, att_mat = encoder(patches)
    class_out = classifier(enc_out)
    vit_out, att_mat_full = vit(x)

    print(f"Input shape: {x.shape}")
    print(f"Patches shape: {patches.shape}")
    # print(f"Attention shape: {atten_weight.shape}")
    print(f"Encoder shape: {enc_out.shape}")
    print(f"Classifier shape: {class_out.shape}")
    print(f"ViT shape: {vit_out.shape}")
    print(f"Attention Matrix shape: {att_mat.shape}")
    print(f"Attention Matrix Full shape: {att_mat_full.shape}")

if __name__ == "__main__":
    test(n_channels=3, embed_dim=32, n_layers=6, image_size=32, 
         patch_size=4, n_attention_heads=4, forward_mul=2, n_classes=10)