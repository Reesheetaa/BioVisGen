"""Generate binary masks for detected text using EasyOCR."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
import easyocr

class TextMaskGenerator:
    def __init__(self, languages=None, padding=6, min_confidence=0.15):
        self.languages = languages or ["en"]
        self.padding = int(padding)
        self.min_confidence = float(min_confidence)
        self.reader = None

    def _get_reader(self):
        if self.reader is None:
            self.reader = easyocr.Reader(self.languages, gpu=torch.cuda.is_available())
        return self.reader

    def generate(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        detections = self._get_reader().readtext(rgb)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = image.size
        for box, text, confidence in detections:
            if float(confidence) < self.min_confidence:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x0 = max(0, int(min(xs)) - self.padding)
            y0 = max(0, int(min(ys)) - self.padding)
            x1 = min(width - 1, int(max(xs)) + self.padding)
            y1 = min(height - 1, int(max(ys)) + self.padding)
            draw.rectangle((x0, y0, x1, y1), fill=255)
        return mask
