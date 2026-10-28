"""Model loading and batched emotion prediction helpers."""
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model import build_model

DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def make_transform(model_name):
    if model_name == "cnn":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def load_predictor(model_name, weights_path=None, device=None):
    if model_name not in ("cnn", "transfer"):
        raise ValueError("model_name must be 'cnn' or 'transfer'")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(weights_path or f"weights/{model_name}_best.pth")
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch
        checkpoint = torch.load(path, map_location=device)
    model = build_model(model_name, pretrained=False).to(device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint.get("classes", DEFAULT_CLASSES), make_transform(model_name), device

def predict_pil_batch(model, images, transform, classes, device):
    """Predict a list of PIL images in one model call; returns (label, confidence) pairs."""
    if not images:
        return []
    batch = torch.stack([transform(im.convert("RGB")) for im in images]).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(batch), dim=1)
        confs, indices = probs.max(dim=1)
    return [(classes[int(i)], float(c)) for i, c in zip(indices.cpu(), confs.cpu())]
