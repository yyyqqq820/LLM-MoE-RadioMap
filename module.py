import os
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from einops import rearrange



# 1. Cross-Attention Fusion Module


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class ImgCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(context_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(context_dim, heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, query_dim)

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(q.shape[0], q.shape[1], self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], k.shape[1], self.heads, -1).permute(0, 2, 3, 1)
        v = v.view(v.shape[0], v.shape[1], self.heads, -1).permute(0, 2, 1, 3)

        sim = torch.matmul(q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class AdaptedTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head):
        super().__init__()
        self.attn = ImgCrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=n_heads,
            dim_head=d_head
        )
        self.ff = FeedForward(dim)

    def forward(self, x, context):
        x = self.attn(x, context=context) + x
        x = self.ff(x) + x
        return x


class AdaptedSpatialTransformer(nn.Module):
    def __init__(self, dim, n_heads=4, d_head=16, depth=1, image_channels=3):
        super().__init__()
        self.dim = dim

        # Image condition encoding
        self.img_norm = nn.BatchNorm2d(image_channels)
        self.img_proj_in = nn.Conv2d(image_channels, dim, kernel_size=1, stride=1, padding=0)

        # Sequence dimension LayerNorm
        self.context_norm = nn.LayerNorm(dim)

        # Semantic input projection
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim)

        self.transformer_blocks = nn.ModuleList([
            AdaptedTransformerBlock(dim, n_heads, d_head) for _ in range(depth)
        ])
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x, context_image):
        # Semantic layer preprocessing
        x = self.norm(x)
        x = self.proj_in(x)

        # Image layer preprocessing
        b, c, h, w = context_image.shape
        context = self.img_norm(context_image)
        context = self.img_proj_in(context)
        context = rearrange(context, 'b c h w -> b (h w) c')

        context = self.context_norm(context)

        # Cross-modal attention
        for block in self.transformer_blocks:
            x = block(x, context)

        x = self.proj_out(x)
        return x



# 2. Basic Components


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x_nodes, adj_dense):
        msg = torch.einsum('n m, b m c -> b n c', adj_dense, x_nodes)
        out = self.linear(msg)
        out = self.norm(out)
        return self.act(out)


class ResDNNBottleneck(nn.Module):
    def __init__(self, dim=128, expansion=2):
        super().__init__()
        mid_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim), nn.GELU(),
            nn.Conv2d(mid_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class RadioResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, pool):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool, stride=pool) if pool > 1 else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return self.pool(out)


def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# 3. Heterogeneous Expert Pool

class ExpertResRadioUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.inputs = in_channels
        self.layer0 = RadioResBlock(in_channels, 32, 3, 1, 1)
        self.layer1 = RadioResBlock(32, 64, 5, 2, 2)
        self.layer2 = RadioResBlock(64, 128, 5, 2, 2)
        self.layer3 = RadioResBlock(128, 256, 5, 2, 2)
        self.layer4 = RadioResBlock(256, 512, 5, 2, 2)

        self.conv_up3 = convreluT(512, 256, 4, 1)
        self.conv_up2 = convreluT(256 + 256, 128, 4, 1)
        self.conv_up1 = convreluT(128 + 128, 64, 4, 1)
        self.conv_up0 = convreluT(64 + 64, 32, 4, 1)

        self.conv_up00 = RadioResBlock(32 + 32, out_channels, 3, 1, 1)

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        u3 = self.conv_up3(l4)
        u3 = torch.cat([u3, l3], dim=1)

        u2 = self.conv_up2(u3)
        u2 = torch.cat([u2, l2], dim=1)

        u1 = self.conv_up1(u2)
        u1 = torch.cat([u1, l1], dim=1)

        u0 = self.conv_up0(u1)
        u0 = torch.cat([u0, l0], dim=1)

        return self.conv_up00(u0)


class ExpertTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, img_size=128, patch_size=16, depth=4, heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True, activation='gelu',
            norm_first=True
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pH, pW = H // self.patch_size, W // self.patch_size
        patches = self.patch_embed(x)
        seq = patches.flatten(2).transpose(1, 2) + self.pos_embed
        seq = self.transformer_block(seq)
        grid_out = seq.transpose(1, 2).contiguous().view(B, -1, pH, pW)
        return self.decoder(grid_out)


class ExpertDNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, base_dim=128):
        super().__init__()
        coord_channels = in_channels + 2 + 2
        self.stem = nn.Sequential(
            nn.Conv2d(coord_channels, base_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_dim), nn.GELU()
        )
        self.res_blocks = nn.Sequential(*[ResDNNBottleneck(dim=base_dim, expansion=2) for _ in range(6)])
        self.head = nn.Sequential(
            nn.Conv2d(base_dim, out_channels, kernel_size=1)
        )

    def forward(self, x, tx_pos, global_coords):
        B, C, H, W = x.shape
        tx_coords = tx_pos.view(B, 2, 1, 1).expand(B, 2, H, W)
        x_with_coords = torch.cat([x, global_coords, tx_coords], dim=1)
        out = self.stem(x_with_coords)
        out = self.res_blocks(out)
        return self.head(out)


class ExpertGNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, hidden_dim=128, grid_size=32, radius=4.0):
        super().__init__()
        self.N = grid_size * grid_size

        self.node_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )

        adj_dense = self._build_dense_adjacency(grid_size, radius)
        self.register_buffer('adj_dense', adj_dense)

        self.gnn1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gnn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gnn3 = GraphConvLayer(hidden_dim, hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_channels),
            nn.LayerNorm(out_channels)
        )

    def _build_dense_adjacency(self, grid_size, radius):
        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        coords = torch.stack([y.flatten(), x.flatten()], dim=1).float()

        dist = torch.cdist(coords, coords)
        adj = (dist <= radius).float()

        degree = adj.sum(dim=-1, keepdim=True)
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(0, 1)

        return norm_adj

    def forward(self, x):
        B, C, H, W = x.shape
        x_nodes = x.view(B, C, -1).transpose(1, 2).contiguous()

        h0 = self.node_embed(x_nodes)
        h1 = self.gnn1(h0, self.adj_dense)
        h2 = self.gnn2(h1, self.adj_dense)
        h3 = self.gnn3(h2, self.adj_dense)

        out_nodes = h3 + h0
        out_nodes = self.out_proj(out_nodes)
        out_img = out_nodes.transpose(1, 2).contiguous().view(B, 64, H, W)
        return out_img


# 4. MoE Framework and Gating Network

class PropagationEnvironmentAugmentation(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cls_embedding = F.normalize(cls_embedding, p=2, dim=1)

        text_embedding = self.mlp(cls_embedding)
        return text_embedding


class GatingNetwork(nn.Module):
    def __init__(self, in_channels=3, num_experts=4, hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts

        # Text path
        self.text_encoder = PropagationEnvironmentAugmentation(hidden_dim=hidden_dim)

        # Image and fusion path
        self.spatial_transformer = AdaptedSpatialTransformer(
            dim=hidden_dim, n_heads=4, d_head=16, depth=1, image_channels=in_channels
        )

        # Router
        self.router = nn.Linear(hidden_dim, num_experts)
        nn.init.zeros_(self.router.weight)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def forward(self, x, input_ids, attention_mask):
        txt_feat = self.text_encoder(input_ids, attention_mask).unsqueeze(1)
        fused_feat = self.spatial_transformer(x=txt_feat, context_image=x)
        logits = self.router(fused_feat.squeeze(1))

        alpha = F.softmax(logits, dim=1)

        # Calculate auxiliary loss for load balancing
        mean_alpha = alpha.mean(dim=0)
        aux_loss = (mean_alpha ** 2).sum() * self.num_experts - 1.0

        return alpha, aux_loss


class MoESpectrumNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, hidden_dim=64):
        super().__init__()
        self.gating_network = GatingNetwork(in_channels=n_channels, num_experts=4, hidden_dim=hidden_dim)

        self.experts = nn.ModuleList([
            ExpertResRadioUNet(n_channels, hidden_dim),
            ExpertTransformer(n_channels, hidden_dim),
            ExpertDNN(n_channels, hidden_dim),
            ExpertGNN(n_channels, hidden_dim)
        ])

        self.decoder = nn.Sequential(
            RadioResBlock(hidden_dim, 64, kernel=3, padding=1, pool=1),
            RadioResBlock(64, 32, kernel=3, padding=1, pool=1),
            RadioResBlock(32, 16, kernel=3, padding=1, pool=1),
            RadioResBlock(16, 8, kernel=3, padding=1, pool=1),
            nn.Conv2d(8, n_classes, kernel_size=1)
        )

    def forward(self, x, input_ids, attention_mask, tx_pos, global_coords):
        weights, aux_loss = self.gating_network(x, input_ids, attention_mask)

        out_agg = torch.zeros(x.size(0), 64, x.size(2), x.size(3), device=x.device)
        for i, expert in enumerate(self.experts):
            w = weights[:, i].view(-1, 1, 1, 1)
            if isinstance(expert, ExpertDNN):
                out_agg += w * expert(x, tx_pos, global_coords)
            else:
                out_agg += w * expert(x)

        logits = self.decoder(out_agg)

        return logits, weights, aux_loss