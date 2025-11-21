import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from llama.llama_adapter import LLaMA_adapter
from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.models import resnet18


# ================================================================
# 1. Load Radar (.mat) file → Tensor [1,1,64,64]
# ================================================================
def load_radar(path):
    data = loadmat(path, simplify_cells=True)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    arr = data[key]

    # Convert complex → magnitude
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    ra = np.array(arr, dtype=np.float32)

    # Ensure 2D
    if ra.ndim == 3:
        ra = ra.mean(axis=0)
    elif ra.ndim > 3:
        ra = ra.reshape(ra.shape[-2], ra.shape[-1])

    # Normalize
    ra = (ra - ra.min()) / (ra.max() - ra.min() + 1e-8)

    # To tensor [1,1,H,W]
    radar = torch.from_numpy(ra).unsqueeze(0).unsqueeze(0)

    # Resize to 64×64
    radar = F.interpolate(
        radar,
        size=(64, 64),
        mode="bilinear",
        align_corners=False
    )
    return radar  # -> [1,1,64,64]


# ================================================================
# 2. Radar ResNet Encoder → 4096-dim visual token
# ================================================================
class RadarResNetEncoder(nn.Module):
    def __init__(self, out_dim=4096):
        super().__init__()
        self.backbone = resnet18(weights=None)

        # Change first conv layer to accept 1-channel radar
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final classifier
        self.backbone.fc = nn.Identity()   # output = 512-dim

        # Project 512 → out_dim (e.g., 4096 for LLaMA)
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        feat = self.backbone(x)        # [B, 512]
        z = self.proj(feat)            # [B, 4096]
        return z


# ================================================================
# 3. Convert radar embedding → LLaMA tokens
# ================================================================
def radar_to_tokens(embed):
    # embed: [B,4096]
    return embed.unsqueeze(1)  # → [B,1,4096]


# ================================================================
# 4. Load LLaMA-Adapter model
# ================================================================
base_dir = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/LLaMA2-7B/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

llama_ckpt_dir = os.path.join(base_dir, "7B")
llama_tokenizer_path = os.path.join(base_dir, "tokenizer.model")

checkpoint_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/llama_adapter_v2_multimodal7b/output/checkpoint-14.pth"

radar_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/unit1/radar1/data_8085_13-37-30.259000.mat"

prompt = "Are there any vehicle or pedestrian in the scene?"

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load LLaMA-Adapter
model = LLaMA_adapter(llama_ckpt_dir, llama_tokenizer_path)
state = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(state["model"], strict=False)
model.to(device).eval()
print("Loaded fine-tuned LLaMA-Adapter model")


# ================================================================
# 5. Prepare Radar Tokens for LLaMA
# ================================================================
radar_tensor = load_radar(radar_path).to(device)  # [1,1,64,64]

# Radar Encoder
radar_encoder = RadarResNetEncoder(out_dim=4096).to(device)

with torch.no_grad():
    radar_embed = radar_encoder(radar_tensor)   # [1,4096]
    radar_tokens = radar_to_tokens(radar_embed) # [1,1,4096]


# ================================================================
# 6. Run Inference with LLaMA Adapter (Radar Only)
# ================================================================
with torch.no_grad():
    result = model.generate(
        radar=radar_tokens,
        imgs=None,               # IGNORE RGB
        prompts=[prompt]
    )[0]

print("\nGenerated answer:")
print(result)


# ================================================================
# 7. Visualize Radar Heatmap
# ================================================================
radar_np = radar_tensor.squeeze().cpu().numpy()

plt.figure(figsize=(5,5))
plt.imshow(radar_np, cmap="jet")
plt.title("Radar Heatmap (64×64)")
plt.colorbar()
plt.axis("off")
plt.show()
