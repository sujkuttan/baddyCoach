import numpy as np

STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]


class BSTClassifier:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.classes = STROKE_CLASSES
        if model_path:
            import torch
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            idx = np.random.randint(len(self.classes))
            return self.classes[idx], 0.8

        import torch
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])
