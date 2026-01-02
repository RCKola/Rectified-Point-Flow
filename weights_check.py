import torch

path = "./output/ALPS_RPF_base/epoch-139.ckpt"
checkpoint = torch.load(path, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)

found_nan = False

for name, param in state_dict.items():
    if torch.isnan(param).any():
        print(f"❌ NaN found in layer: {name}")
        found_nan = True
    elif torch.isinf(param).any():
        print(f"⚠️ Inf found in layer: {name}")
        found_nan = True

if not found_nan:
    print("✅ No NaNs or Infs found in checkpoint weights.")