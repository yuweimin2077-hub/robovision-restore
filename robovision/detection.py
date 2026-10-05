"""A fixed pretrained detector. Never train or tune on test images."""
import numpy as np


class Detector:
    def __init__(self, device="auto"):
        import torch
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
        self.torch = torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.names = weights.meta["categories"]
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(self.device)

    def __call__(self, image):
        tensor = self.torch.from_numpy(np.clip(image, 0, 1).astype(np.float32).transpose(2, 0, 1).copy()).to(self.device)
        with self.torch.inference_mode():
            prediction = self.model([tensor])[0]
        return {key: prediction[key].detach().cpu().numpy() for key in ("boxes", "labels", "scores")}
