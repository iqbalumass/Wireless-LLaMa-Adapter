from scipy.io import loadmat
import torch

def load_radar_mat(path, field="data"):
    """Load radar .mat data as torch tensor."""
    mat = loadmat(path, simplify_cells=True)
    if field not in mat:
        raise KeyError(f"{field} not in {path}. Available: {list(mat.keys())}")
    arr = mat[field]  # shape (4, 256, 128)
    arr = torch.tensor(arr, dtype=torch.float32)
    return arr