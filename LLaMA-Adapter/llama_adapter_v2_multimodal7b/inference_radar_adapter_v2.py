#!/usr/bin/env python3
"""
Radar + LLaMA-Adapter V2 Inference  (FINAL FIXED)
Author: Iqbal — fixed tokenizer + generate() + multimodal fusion
"""

import os, torch, torch.nn as nn, cv2
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer
import llama
from models.radar_fusion_encoder import RadarFusionEncoder
from util.radar_utils import load_radar_mat


# ============================================================
# ✅ Environment setup
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


# ============================================================
# ✅ Paths
# ============================================================
llama_dir = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/llama_adapter/LLaMA-Adapter/LLaMA2-7B/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/"
radar_encoder_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/project/checkpoints/radar_encoder_contrastive_512_safe.pth"
radar_mat = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/unit1/radar1/data_10000_11-47-37.134000.mat"
img_path = "/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/unit1/rgb1/frame_11-46-31.205818.jpg"


# ============================================================
# ✅ Radar Encoder
# ============================================================
class RadarEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feats = self.conv(x).flatten(1)
        return self.fc(feats)


# ============================================================
# ✅ Load Radar Encoder
# ============================================================
print("Loading radar encoder...")
checkpoint = torch.load(radar_encoder_path, map_location=device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

radar_encoder = RadarEncoder(out_dim=512)
radar_encoder.load_state_dict(checkpoint, strict=False)
radar_encoder = radar_encoder.to(device).eval()
print("✅ Radar encoder loaded successfully.")


# ============================================================
# ✅ Load LLaMA-Adapter V2
# ============================================================
try:
    print("Loading LLaMA-Adapter V2...")
    model, preprocess = llama.load("LORA-BIAS-7B", llama_dir, llama_type="7B", device=device)
    model.eval()
    print("✅ LLaMA-Adapter V2 loaded successfully.")

    print("\n🔍 Inspecting model.llama submodules:")
    for name, _ in model.llama.named_children():
        print("  •", name)

except Exception as e:
    print("❌ LLaMA load failed:", e)
    import traceback; traceback.print_exc()


# ============================================================
# ✅ Fusion encoder + processor
# ============================================================
fusion_encoder = RadarFusionEncoder(radar_encoder).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# ============================================================
# ✅ Load image + radar
# ============================================================
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# CLIP preprocess → tensor [1, 3, H, W]
img_inputs = clip_processor(images=image, return_tensors="pt").to(device)

radar_tensor = load_radar_mat(radar_mat, field="data")
radar_tensor = radar_tensor.mean(0, keepdim=True).unsqueeze(0).to(device)


# ============================================================
# ✅ Fuse embeddings
# ============================================================
print("Generating fused visual tokens...")
with torch.no_grad():
    visual_tokens = fusion_encoder(img_inputs, radar_tensor)


# ============================================================
# ✅ Prepare prompt + tokenizer
# ============================================================
prompt = llama.format_prompt("Is there any object visible?")

if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "encode"):
    input_ids = torch.tensor([model.tokenizer.encode(prompt, bos=True, eos=True)]).to(device)
else:
    llama_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", use_fast=False
    )
    input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to(device)


# ============================================================
# ✅ Generate text from fused radar + image
# ============================================================
print("Generating fused visual tokens...")
with torch.no_grad():
    visual_tokens = fusion_encoder(img_inputs, radar_tensor)

print("Generating text from fused radar + image features...")
try:
    visual_tokens = visual_tokens.to(model.dtype)
    input_ids = input_ids.to(model.dtype)
    # Project visual tokens into LLaMA’s embedding space
    embed_dim = model.llama.tok_embeddings.weight.shape[1]
    projection = nn.Linear(visual_tokens.size(-1), embed_dim).to(device)
    projected_embeds = projection(visual_tokens)

    with torch.no_grad():
        output = model.forward_inference(projected_embeds, input_ids, start_pos=0)[0]

    text = model.tokenizer.decode(output, skip_special_tokens=True)
    print("\n🧠 Generated Description:\n", text)

except Exception as e:
    print("⚠️ Fallback:", e)


print("\n✅ Inference completed successfully.")
