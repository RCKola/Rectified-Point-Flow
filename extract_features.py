"""Extract features from pointclouds and visualize PCA."""

from functools import partial
from typing import Dict
import numpy as np
import torch
import rectified_point_flow.encoder.concerto.concerto as concerto
import rectified_point_flow.data as data

try:
    import flash_attn
except ImportError:
    flash_attn = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color

def extract_point_features(encoder, batch: Dict[str, torch.Tensor]) -> tuple:
    """Extract point features using the encoder."""
    pointclouds = batch["pointclouds"]                          # (B, N, 3)
    normals = batch["pointclouds_normals"]                      # (B, N, 3)
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
        print("Point encoder output keys:", point.keys())
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

def visualize_features(point_features: torch.Tensor, point_data: Dict[str, torch.Tensor]):
    """Visualize point features using PCA coloring."""
    pca_color = get_pca_color(point_features, brightness=1.2, center=True)
    batched_coord = point_data["coord"].clone()
    batched_coord[:, 0] += point_data["batch"] * 8.0

    from rectified_point_flow.visualizer import VisualizationCallback
    vis_cb = VisualizationCallback(save_dir="outputs/concerto_feature_viz")
    vis_cb._save_sample_images(
        points = batched_coord,
        colors = pca_color,
        sample_name = "concerto_features"
    ) 


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract features from concerto.')
    parser.add_argument('-d','--data-path', type=str, required=False, default=None,
                        help='Path to the point cloud data file.')
    args = parser.parse_args()

    concerto.utils.set_seed(42)
    if flash_attn is not None:
        model = concerto.load("concerto_large", repo_id="Pointcept/Concerto").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = concerto.load(
            "concerto_large", repo_id="Pointcept/Concerto", custom_config=custom_config
        ).to(device)
    
    model.enc_mode = True
    print("Model loaded.")
    print("Model expected input dim", model.embedding.in_channels)
    
    # Load data
    datamodule = data.PointCloudDataModule(
        data_root=f"{args.data_path}/datasets",
        dataset_names=["ikea"],
        num_points_to_sample=2000,
        batch_size=2,
        num_workers=4
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    sample = next(iter(loader))
    print("Sample keys:", sample.keys())
    print("Sample pointclouds shape:", sample["pointclouds"].shape)
    print("Sample num parts", sample["num_parts"])
    print("Sample loaded.")
    
    # Extract features
    point_features, point_data, _ = extract_point_features(model, sample)

    print("Feature extraction completed.")
    print("Extracted point features shape:", point_features.shape)
    print("Extracted features:", point_features)
    print("Point coordinates keys:", point_data.keys())
    print("Point coordinates shape:", point_data["coord"].shape)
    print("Point batch shape:", point_data["batch"].shape)
    print("Point batch:", point_data["batch"])

    # Save features and coordinates
    np.savez(
        "outputs/ikea_sample_features.npz",
        point_features=point_features.cpu().detach().numpy(),
        point_coords=point_data["coord"].cpu().detach().numpy(),
        index=sample["index"],
        name=sample["name"],
        dataset=sample["dataset_name"],
    )


if __name__ == "__main__":
    main()