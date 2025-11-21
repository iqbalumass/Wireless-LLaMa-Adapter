# data/radar_dataset.py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F

class RadarImageTextDataset(Dataset):
    def __init__(self, ann_file, transform, tokenizer):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def _load_radar(self, path):
        data = loadmat(path, simplify_cells=True)
        key = [k for k in data.keys() if not k.startswith("__")][0]
    
        arr = data[key]
    
        # 1) handle complex -> use magnitude
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
    
        ra = np.array(arr, dtype=np.float32)
    
        # 2) collapse extra dims to 2D map
        if ra.ndim == 3:
            ra = ra.mean(axis=0)
        elif ra.ndim > 3:
            ra = ra.reshape(ra.shape[-2], ra.shape[-1])
    
        assert ra.ndim == 2, f"Expected 2D radar map, got shape {ra.shape}"
    
        # 3) normalize to [0,1]
        ra = (ra - ra.min()) / (ra.max() - ra.min() + 1e-8)
    
        # 4) to torch CHW and resize to 64x64
        radar = torch.from_numpy(ra).unsqueeze(0)  # [1,H,W]
        radar = F.interpolate(
            radar.unsqueeze(0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
    
        return radar

    def __getitem__(self, i):
        it = self.ann[i]
        img = Image.open(it["image"]).convert("RGB")
        img = self.transform(img)
        radar = self._load_radar(it["radar"])
        text_input  = it["conversations"][0]["value"]
        text_target = it["conversations"][1]["value"]
        full_text = f"{text_input}\n{text_target}"
        print(full_text)
        # Convert to tensor right here
        tokens = torch.tensor(self.tokenizer.encode(full_text, bos=True, eos=True), dtype=torch.long)
    
        return {"image": img, "radar": radar, "tokens": tokens, "labels": tokens.clone()}


# ------------------------------
# Padding collate function (GLOBAL)
# ------------------------------
def collate_fn(batch):
    """Pads variable-length text tokens for batching."""
    images = torch.stack([b["image"] for b in batch])
    radars = torch.stack([b["radar"] for b in batch])

    tokens = [b["tokens"] for b in batch]
    labels = [b["labels"] for b in batch]
    max_len = max(t.size(0) for t in tokens)

    padded_tokens = torch.zeros(len(tokens), max_len, dtype=torch.long)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)

    for i in range(len(tokens)):
        L = tokens[i].size(0)
        padded_tokens[i, :L] = tokens[i]
        padded_labels[i, :L] = labels[i]

    return {
        "image": images,
        "radar": radars,
        "tokens": padded_tokens,
        "labels": padded_labels,
    }
