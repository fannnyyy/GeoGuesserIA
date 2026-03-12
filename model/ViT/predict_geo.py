from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

"""
readme : 

changer le chemin du checkpoint pour l emplacement de vit_geo_best.pt

necessaire : 
pip install torch torchvision pillow

pour tester sur une image : 
python predict_geo.py --image example.jpg --checkpoint vit_geo_best.pt
en changeant l image quand on lance le job : sbatch run_predict_geo.slurm
"""





CHECKPOINT_PATH = Path("checkpoints/vit_geo_best.pt")


def normalize_vec(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def unitvec_to_latlon(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = normalize_vec(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    lat = torch.asin(z)
    lon = torch.atan2(y, x)
    return torch.rad2deg(lat), torch.rad2deg(lon)


class ViTGeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vit_b_16(weights=None)
        self.net.heads.head = nn.Linear(self.net.heads.head.in_features, 3)

    def forward(self, x):
        return self.net(x)


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    model = ViTGeo().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )


@torch.no_grad()
def predict_image(model: nn.Module, image_path: Path, device: torch.device) -> tuple[float, float]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

    image = Image.open(image_path).convert("RGB")
    transform = build_transform()
    x = transform(image).unsqueeze(0).to(device)  # shape (1, 3, 224, 224)

    pred_vec = model(x)
    pred_vec = normalize_vec(pred_vec)

    lat, lon = unitvec_to_latlon(pred_vec)
    return float(lat.item()), float(lon.item())


def parse_args():
    parser = argparse.ArgumentParser(description="Predict latitude/longitude from an image using ViTGeo.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to the model checkpoint (.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)

    model = load_model(checkpoint_path, device)
    lat, lon = predict_image(model, image_path, device)

    print(f"Predicted latitude : {lat:.6f}")
    print(f"Predicted longitude: {lon:.6f}")


if __name__ == "__main__":
    main()
    
  