import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class TrackNetV3Model(nn.Module):
    def __init__(self, in_channels=27, num_classes=8):
        super().__init__()
        # Encoder
        self.down_block_1 = nn.ModuleDict({
            'conv_1': SingleConv(in_channels, 64),
            'conv_2': SingleConv(64, 64),
        })
        self.down_block_2 = nn.ModuleDict({
            'conv_1': SingleConv(64, 128),
            'conv_2': SingleConv(128, 128),
        })
        self.down_block_3 = nn.ModuleDict({
            'conv_1': SingleConv(128, 256),
            'conv_2': SingleConv(256, 256),
            'conv_3': SingleConv(256, 256),
        })

        # Bottleneck
        self.bottleneck = nn.ModuleDict({
            'conv_1': SingleConv(256, 512),
            'conv_2': SingleConv(512, 512),
            'conv_3': SingleConv(512, 512),
        })

        # Decoder
        self.up_block_1 = nn.ModuleDict({
            'conv_1': SingleConv(512 + 256, 256),
            'conv_2': SingleConv(256, 256),
            'conv_3': SingleConv(256, 256),
        })
        self.up_block_2 = nn.ModuleDict({
            'conv_1': SingleConv(256 + 128, 128),
            'conv_2': SingleConv(128, 128),
        })
        self.up_block_3 = nn.ModuleDict({
            'conv_1': SingleConv(128 + 64, 64),
            'conv_2': SingleConv(64, 64),
        })

        self.predictor = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down_block_1['conv_1'](x)
        d1 = self.down_block_1['conv_2'](d1)
        d1_pool = nn.functional.max_pool2d(d1, 2)

        d2 = self.down_block_2['conv_1'](d1_pool)
        d2 = self.down_block_2['conv_2'](d2)
        d2_pool = nn.functional.max_pool2d(d2, 2)

        d3 = self.down_block_3['conv_1'](d2_pool)
        d3 = self.down_block_3['conv_2'](d3)
        d3 = self.down_block_3['conv_3'](d3)
        d3_pool = nn.functional.max_pool2d(d3, 2)

        # Bottleneck
        b = self.bottleneck['conv_1'](d3_pool)
        b = self.bottleneck['conv_2'](b)
        b = self.bottleneck['conv_3'](b)

        # Decoder
        b_up = nn.functional.interpolate(b, size=d3.shape[2:], mode='bilinear', align_corners=True)
        u1 = torch.cat([b_up, d3], dim=1)
        u1 = self.up_block_1['conv_1'](u1)
        u1 = self.up_block_1['conv_2'](u1)
        u1 = self.up_block_1['conv_3'](u1)

        u1_up = nn.functional.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=True)
        u2 = torch.cat([u1_up, d2], dim=1)
        u2 = self.up_block_2['conv_1'](u2)
        u2 = self.up_block_2['conv_2'](u2)

        u2_up = nn.functional.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=True)
        u3 = torch.cat([u2_up, d1], dim=1)
        u3 = self.up_block_3['conv_1'](u3)
        u3 = self.up_block_3['conv_2'](u3)

        return self.predictor(u3)


class TrackNetV3:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.input_height = 288
        self.input_width = 512

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.model = TrackNetV3Model()
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()

    def _preprocess(self, frames: list[np.ndarray]) -> np.ndarray:
        import cv2
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            processed.append(normalized)
        # Stack frames: (num_frames, H, W, C) -> flatten to (num_frames*C, H, W)
        batch = np.stack(processed)  # (N, H, W, 3)
        batch = batch.reshape(-1, self.input_height, self.input_width)  # (N*3, H, W)
        batch = batch[np.newaxis, ...]  # (1, N*3, H, W)
        return torch.from_numpy(batch).float().to(self.device)

    def _postprocess(self, output: np.ndarray, original_width: int, original_height: int) -> dict:
        # Apply sigmoid to convert logits to probabilities
        heatmap = 1 / (1 + np.exp(-output))
        y_idx, x_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        confidence = float(heatmap.max())
        x = x_idx * original_width / self.input_width
        y = y_idx * original_height / self.input_height
        return {"x": float(x), "y": float(y), "confidence": confidence}

    def predict(self, frames: list[np.ndarray], original_size: tuple | None = None) -> list[dict]:
        if self.model is None or len(frames) < 3:
            h = frames[0].shape[0] if frames else 720
            w = frames[0].shape[1] if frames else 1280
            return [{"x": 0, "y": 0, "confidence": 0}]

        original_width = original_size[0] if original_size else frames[0].shape[1]
        original_height = original_size[1] if original_size else frames[0].shape[0]

        # Use last 9 frames if available, otherwise pad with copies
        if len(frames) >= 9:
            input_frames = frames[-9:]
        else:
            input_frames = frames + [frames[-1]] * (9 - len(frames))

        tensor = self._preprocess(input_frames)

        with torch.no_grad():
            output = self.model(tensor)

        heatmap = output.cpu().numpy()[0, 0]
        return [self._postprocess(heatmap, original_width, original_height)]

    def predict_batch(self, frames: list[np.ndarray], batch_size: int = 3, original_size: tuple | None = None) -> list[dict]:
        if len(frames) < 3:
            return [{"x": 0, "y": 0, "confidence": 0} for _ in frames]

        results = []
        for i in range(2, len(frames)):
            window = frames[max(0, i-8):i+1]
            pred = self.predict(window, original_size)
            results.append(pred[0])
        return results
