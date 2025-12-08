"""Extract features from pointclouds and visualize PCA."""

from typing import Dict, Optional
import numpy as np
import torch
import rectified_point_flow.encoder.concerto.concerto as concerto
import rectified_point_flow.data as data

import logging
from pathlib import Path
import os
import warnings

import hydra
import lightning as L
from omegaconf import DictConfig

from rectified_point_flow.utils import load_checkpoint_for_module, download_rfp_checkpoint, print_eval_table

logger = logging.getLogger("ExtractFeatures")
warnings.filterwarnings("ignore", module="lightning")
warnings.filterwarnings("ignore", category=FutureWarning)


def extract_point_features(encoder, batch: Dict[str, torch.Tensor]) -> tuple:
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

def visualize_features(features: torch.Tensor, coords: torch.Tensor, save_dir: str, id: int = 0):
    """Visualize point features using PCA coloring."""
    pca_color = get_pca_color(features, brightness=1.2, center=True)

    from rectified_point_flow.visualizer import VisualizationCallback
    vis_cb = VisualizationCallback(save_dir=save_dir, image_size=2048)
    vis_cb.setup(None, None, 'test')

    vis_cb._save_sample_images(
        points = coords,
        colors = pca_color,
        sample_name = f"concerto_features_{id:02d}.png"
    ) 

def run_visuals(load_path: str):
    data = np.load(load_path)
    point_features = torch.tensor(data["point_features"]).to(device)
    point_coords = torch.tensor(data["point_coords"]).to(device)

    batched_coords = point_coords.reshape(2, -1, 3)

    print("Loaded point features shape:", point_features.shape)
    print("Loaded point coordinates shape:", batched_coords.shape)

    for i, coords in enumerate(batched_coords):
        visualize_features(
            features=point_features[i],
            coords=coords,
            save_dir="outputs/concerto_features",
            id=i
        )

def load_concerto():
    try:
        import flash_attn
    except ImportError:
        flash_attn = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device:", device)

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
    
    logger.info(f"Concerto loaded. Expecting input dim: {model.embedding.in_channels}")
    return model


def setup(cfg: DictConfig):
    """Setup inference components."""
    
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        ckpt_path = download_rfp_checkpoint("RPF_base_pretrain_ep600.ckpt", './weights')
    elif not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Please provide a valid checkpoint in the config or via ckpt_path='...' argument")
        exit(1)

    # Seed if set
    seed = cfg.get("seed", None)
    if seed is not None:
        L.seed_everything(seed, workers=True, verbose=False)
        logger.info(f"Seed set to {seed} for overlap prediction")

    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(model, ckpt_path)
    model.eval()
    
    vis_config = cfg.get("visualizer", {})
    callbacks = []
    if vis_config and cfg["visualizer"]["renderer"] != "none":
        vis_callback = hydra.utils.instantiate(vis_config)
        callbacks.append(vis_callback)
    
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
    )
    
    return model, datamodule, trainer


class ConcertoModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = load_concerto()
    
    def forward(self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        point_features, point_data, n_valid_parts = extract_point_features(self.model.embedding, batch)
        output = {
            "point": point_data,
            "n_valid_parts": n_valid_parts,
        }
        return output

    def test_step(self, batch, batch_idx):
        results = self.forward(batch)
        return results


@hydra.main(version_base="1.3", config_path="./config", config_name="RPF_REPA_extract_features")
def main(cfg: DictConfig):
    model, datamodule, trainer = setup(cfg)
    eval_results = trainer.test(
        model=model, 
        datamodule=datamodule, 
        verbose=False,
    )
    # print_eval_table(eval_results, datamodule.dataset_names)
    logger.info("Visualizations saved to:" + str(Path(cfg.get('log_dir')) / "visualizations"))

    # concerto_module = ConcertoModule()
    return
    
    # Load data
    datamodule = data.PointCloudDataModule(
        data_root=f"{args.data_path}/datasets",
        dataset_names=["ikea"],
        num_points_to_sample=2000,
        batch_size=10,
        num_workers=4
    )
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    sample = next(iter(loader))
    print("Sample keys:", sample.keys())
    print("Sample num parts", sample["num_parts"])
    print("Sample loaded.")
    
    # Extract features
    point_features, point_data, _ = extract_point_features(model, sample)
    print("Feature extraction completed.")

    # Save features and coordinates
    filepath = save_dir + "/ikea_sample_features.npz"
    np.savez(
        filepath,
        point_features=point_features.cpu().detach().numpy(),
        point_coords=point_data["coord"].cpu().detach().numpy(),
        index=sample["index"],
        name=sample["name"],
        dataset=sample["dataset_name"],
    )
    print(f"Features saved to {filepath}")

    # Visualize features
    batch_coords = point_data["coord"].reshape(point_features.shape[0], -1, 3)
    for i in range(point_features.shape[0]):
        visualize_features(
            features=point_features[i],
            coords=batch_coords[i],
            save_dir=save_dir,
            id=i
        )


if __name__ == "__main__":
    main()