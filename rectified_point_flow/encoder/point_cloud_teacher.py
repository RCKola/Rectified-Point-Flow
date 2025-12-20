import torch
import torch.nn as nn
import torch.nn.functional as F


def get_graph_feat(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Optimized KNN feature generator from DGCNN"""
    B, N, C = x.shape

    # Compute indices using KNN
    pair_dist = -torch.cdist(x, x, p=2).pow(2)
    idx = pair_dist.topk(k=k, dim=-1)[1]                # (B, N, k)

    x_ = x.unsqueeze(1).expand(-1, N, -1, -1)           # (B, N, C) -> (B, 1, N, C) -> (B, N, N, C)
    idx_ = idx.unsqueeze(-1).expand(-1, -1, -1, C)      # (B, N, k, C)
    feat = torch.gather(x_, dim=2, index=idx_)          # (B, N, k, C)
    center = x.unsqueeze(2)                             # (B, N, 1, C)

    features = torch.empty(B, N, k, 2*C, device=x.device, dtype=x.dtype)
    features[..., :C] = feat - center
    features[..., C:] = center
    return features.permute(0, 3, 1, 2)                 # (B, 2*C, N, k)

class PointCloudTeacher(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        repr_dim: int = 1728, #  Output dimension of Concerto large
        final_mlp_act: nn.Module = nn.SiLU,
        lmbda: float = 0.5,
        loss_type: str = "cosine",
        head_type: str = "edgeconv"
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.repr_dim = repr_dim
        self.final_mlp_act = final_mlp_act
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.head_type = head_type

        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim, self.embed_dim),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim, self.repr_dim)
        ) if self.loss_type != "disp_l2" or self.head_type != "mlp" else nn.Identity()

        self.edgeconv1 = nn.Sequential(
            nn.Conv2d(2 * self.embed_dim, self.repr_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.repr_dim // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.edgeconv2 = nn.Sequential(
            nn.Conv2d(self.repr_dim, self.repr_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.repr_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        if encoder is None and loss_type != "disp_l2":
            from .concerto import concerto
            self.encoder = concerto.load("concerto_large", repo_id="Pointcept/Concerto")
            self._freeze_model(self.encoder)
    
    def edgeconv(self, x: torch.Tensor) -> torch.Tensor:
        x = get_graph_feat(x, k=5)                  # (B, N, C) -> (B, 2C, N, k)
        x1 = self.edgeconv1(x)                      # (B, D/2, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]       # (B, D/2, N)
        x1 = x1.transpose(1, 2)                     # (B, N, D/2)

        x = get_graph_feat(x1, k=5)                 # (B, D, N, k)
        x2 = self.edgeconv2(x)                      # (B, D, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]       # (B, D, N)
        x2 = x2.transpose(1, 2)                     # (B, N, D)
        return x2
            
    def forward(self, interm_repr: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'force':
            interm_repr = F.normalize(interm_repr, p=2, dim=-1)
        if self.head_type == "edgeconv":
            aligned_repr = self.edgeconv(interm_repr)
        elif self.head_type == "mlp":
            aligned_repr = self.mlp_head(interm_repr)
        else:
            raise ValueError(f"Invalid head type: {self.head_type}")
        return aligned_repr

    def extract_features(self, batch: dict) -> tuple:
        """Extract point features using the encoder."""
        pointclouds = batch["pointclouds_gt"]                           # (B, N, 3)
        normals = batch["pointclouds_normals_gt"]                       # (B, N, 3)
        points_per_part = batch["points_per_part"]                      # (B, P)
        B, N, C = pointclouds.shape
        n_valid_parts = points_per_part != 0
        
        # Prepare inputs for encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"

        part_coords = pointclouds.view(-1, C).to(device)
        part_normals = normals.view(-1, C).to(device)
        points_offset = torch.cumsum(points_per_part[n_valid_parts], dim=-1).to(device)
        object_offset = torch.arange(1, B + 1, device=device) * N
        color = torch.zeros_like(part_coords).to(device)  # Dummy color
        grid_size = 0.02

        encoder = self.encoder
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.float16):
                point = encoder({
                    "coord": part_coords,
                    "color": color,
                    "offset": points_offset,
                    "offset_level1": points_offset,                     # Offset to split parts
                    "offset_level0": object_offset,                     # Offset to split objects
                    "feat": torch.cat([part_coords, part_normals, color], dim=-1),
                    "grid_size": torch.tensor(grid_size),
                })
                # dict_keys(['feat', 'coord', 'grid_coord', 'batch', 'color', 'grid_size', 
                # 'pooling_inverse', 'pooling_parent', 'idx_ptr', 'offset', 'order', 
                # 'serialized_depth', 'serialized_code', 'serialized_order', 'serialized_inverse', 
                # 'sparse_shape', 'sparse_conv_feat', 'pad', 'unpad', 'cu_seqlens_key'])
                while "pooling_parent" in point.keys():
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                    point = parent
                point["normal"] = part_normals
                features = point["feat"].reshape(B, N, -1)
        return features, point, n_valid_parts
    
    def get_target(self, data_dict: dict) -> torch.Tensor | None:
        """Retrieve target representation for alignment loss."""
        if data_dict is None or self.loss_type == "disp_l2":
            return None
        features = self.extract_features(data_dict)[0]
        if self.head_type == "edgeconv":
            features = self._spatial_normalize(features)
        return features

    def loss(self, repr_pred: torch.Tensor, repr_t: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss."""
        if repr_t is None or self.encoder is None:
            self.loss_type = "disp_l2"

        if self.loss_type == "cosine":
            loss = 1 - F.cosine_similarity(repr_pred, repr_t, dim=-1).mean()
        elif self.loss_type == "force":
            loss = 1 - F.cosine_similarity(repr_pred, repr_t, dim=-1).mean()
            loss += F.mse_loss(repr_pred, repr_t.clone().detach(), reduction="mean")
        elif self.loss_type == "disp_l2":
            z = torch.flatten(repr_pred, 1)
            dist = F.pdist(z, p=2).pow(2) / z.shape[1]
            tau = 0.5

            # Accounts for zero distance to self
            dist = torch.concat([dist, dist, torch.zeros(z.shape[0]).to(dist.device)]) 

            # Log sum exp trick for numerical stability
            import math
            loss = torch.logsumexp(-dist/tau, dim=0) - math.log(dist.numel())
            return loss
        else:
            raise ValueError(f"Invalid alignment loss type: {self.loss_type}")
        return self.lmbda * loss
    
    def _spatial_normalize(self, x: torch.tensor, gamma: float = 1.0) -> torch.Tensor:
        x = x - gamma * x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    def _freeze_model(self, model: nn.Module):
        model.eval()
        for module in model.modules():
            module.eval()
        for param in model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    teacher = PointCloudTeacher(
        encoder=None,
        embed_dim=512,
        repr_dim=1728,
        final_mlp_act=nn.SiLU,
        lmbda=0.5,
    )
    print(f"PointCloudTeacher with {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M parameters")



