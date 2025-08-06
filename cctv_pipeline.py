import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration
)

# ======= Image Captioning using BLIP =======
class ImageCaptioner:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading BLIP model on {self.device}...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

# ======= CCTV Frame Extractor & YOLOv8 Object + Person Detector =======
class CCTVObjectDetector:
    def __init__(self, video_path, output_dir="output", model_path="yolov8n.pt", frame_interval=30, similarity_threshold=0.85):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.similarity_threshold = similarity_threshold
        self.model = YOLO(model_path)
        self._setup_dirs()

    def _setup_dirs(self):
        os.makedirs(f"{self.output_dir}/frames_unique", exist_ok=True)
        os.makedirs(f"{self.output_dir}/detections", exist_ok=True)

    def extract_unique_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count, saved_count = 0, 0
        prev_gray = None
        unique_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None or ssim(prev_gray, gray) < self.similarity_threshold:
                    fname = f"frame_{saved_count:04d}.jpg"
                    fpath = os.path.join(self.output_dir, "frames_unique", fname)
                    cv2.imwrite(fpath, frame)
                    unique_frames.append((fpath, frame))
                    prev_gray = gray
                    saved_count += 1

            frame_count += 1

        cap.release()
        return unique_frames

    def detect_objects(self, frames):
        results = []
        for fpath, frame in frames:
            result = self.model(frame)[0]
            detections = []

            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                class_id = int(cls)
                class_name = self.model.names[class_id]

                detection = {
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 2),
                    "class": class_name
                }
                detections.append(detection)

                # Draw bounding box
                color = (0, 255, 0) if class_name == "person" else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out_path = os.path.join(self.output_dir, "detections", os.path.basename(fpath))
            cv2.imwrite(out_path, frame)

            results.append({
                "image": os.path.basename(fpath),
                "detections": detections
            })

        return results


# ======= Caption Summarizer (T5/BART) =======
class CaptionSummarizer:
    def __init__(self, model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading summarization model ({model_name}) on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def summarize(self, captions, max_tokens=256):
        full_text = " ".join(captions)
        prompt = f"summarize: {full_text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_tokens, num_beams=4)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
