import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudTeacher(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        repr_dim: int = 1728, #  Output dimension of Concerto large
        final_mlp_act: nn.Module = nn.SiLU,
        lmbda: float = 0.5,
        loss_type: str = "cosine"
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.repr_dim = repr_dim
        self.final_mlp_act = final_mlp_act
        self.lmbda = lmbda
        self.loss_type = loss_type

        self.alignment_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim, self.embed_dim),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim, repr_dim)
        )

        if encoder is None and loss_type != "disp_l2":
            from .concerto import concerto
            self.encoder = concerto.load("concerto_large", repo_id="Pointcept/Concerto")
            self._freeze_model(self.encoder)
            
    def forward(self, interm_repr: torch.Tensor):
        aligned_repr = self.alignment_head(interm_repr)
        return aligned_repr

    def extract_features(self, batch: dict) -> tuple:
        """Extract point features using the encoder."""
        pointclouds = batch["pointclouds_gt"]                          # (B, N, 3)
        normals = batch["pointclouds_normals_gt"]                      # (B, N, 3)
        points_per_part = batch["points_per_part"]                  # (B, P)
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
    
    def get_target(self, data_dict: dict):
        """Retrieve target representation for alignment loss."""
        if data_dict is None or self.loss_type == "disp_l2":
            raise ValueError("No target representation available for the specified loss type.")
        features = self.extract_features(data_dict)[0]
        return features

    def loss(self, repr_pred: torch.Tensor, repr_t: torch.Tensor):
        """Compute alignment loss."""
        if repr_t is None or self.encoder is None:
            self.loss_type = "disp_l2"

        if self.loss_type == "cosine":
            loss = 1 - F.cosine_similarity(repr_pred, repr_t, dim=-1).mean()
        elif self.loss_type == "force":
            loss = 1 - F.cosine_similarity(repr_pred, repr_t, dim=-1).mean()
            loss += F.mse_loss(repr_pred, repr_t, reduction="mean")
        elif self.loss_type == "disp_l2":
            z = repr_pred.reshape(repr_pred.shape[0], -1)
            dist = F.pdist(z, p=2).pow(2) / repr_pred.shape[-1]
            tau = 0.5
            loss = torch.log(torch.exp(-dist/tau).mean())
        else:
            raise ValueError(f"Invalid alignment loss type: {self.loss_type}")
        return self.lmbda * loss
    
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



