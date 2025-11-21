import torch
from PIL import Image
from llama.llama_adapter import LLaMA_adapter
from data.radar_dataset import RadarImageTextDataset
from scipy.io import loadmat
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt       

# --------------------------------------------------------------------
# Helper: radar loader (same as in dataset)
# --------------------------------------------------------------------
def load_radar(path):
    data = loadmat(path, simplify_cells=True)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    arr = data[key]

    # Convert complex → magnitude
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    ra = np.array(arr, dtype=np.float32)
    if ra.ndim == 3:
        ra = ra.mean(axis=0)
    elif ra.ndim > 3:
        ra = ra.reshape(ra.shape[-2], ra.shape[-1])

    ra = (ra - ra.min()) / (ra.max() - ra.min() + 1e-8)
    radar = torch.from_numpy(ra).unsqueeze(0)  # [1,H,W]
    radar = F.interpolate(
        radar.unsqueeze(0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)  # [1,64,64]
    return radar


# --------------------------------------------------------------------
# ✅ Your correct paths
# --------------------------------------------------------------------
base_dir = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/LLaMA2-7B/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

llama_ckpt_dir = os.path.join(base_dir, "7B")
llama_tokenizer_path = os.path.join(base_dir, "tokenizer.model")  

checkpoint_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/llama_adapter_v2_multimodal7b/output/checkpoint-14.pth"

image_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/unit1/rgb1/frame_13-37-28.998361.jpg"
radar_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/unit1/radar1/data_8085_13-37-30.259000.mat"

prompt = "Are there any vehicle or pedestrian in the scene?"

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------
# Load fine-tuned model
# --------------------------------------------------------------------
model = LLaMA_adapter(llama_ckpt_dir, llama_tokenizer_path)
state = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(state["model"], strict=False)
model.to(device).eval()
print("Loaded fine-tuned LLaMA-Adapter-Radar model")


# --------------------------------------------------------------------
# Prepare inputs
# --------------------------------------------------------------------
img = Image.open(image_path).convert("RGB")
img_tensor = model.clip_transform(img).unsqueeze(0).to(device)
radar_tensor = load_radar(radar_path).unsqueeze(0).to(device)


# --------------------------------------------------------------------
# Generate caption
# --------------------------------------------------------------------
with torch.no_grad():
    result = model.generate(imgs=img_tensor, prompts=[prompt], radar=radar_tensor)[0]

print("\n Generated caption:")
print(result)


# --------------------------------------------------------------------
#  Show Image + Radar
# --------------------------------------------------------------------
radar_np = radar_tensor.squeeze().cpu().numpy()   # -> (64,64)

plt.figure(figsize=(12,5))

# --- RGB Image ---
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("RGB Image")
plt.axis("off")

# --- Radar Map ---
plt.subplot(1,2,2)
plt.imshow(radar_np, cmap="jet")
plt.title("Radar Heatmap (64×64)")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()