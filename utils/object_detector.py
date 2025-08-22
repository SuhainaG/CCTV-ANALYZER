import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import List, Dict, Tuple
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_path: str = 'yolov8x.pt', confidence_threshold: float = 0.25):
        """
        Initialize ObjectDetector with YOLOv8x model
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        
        try:
            # Load YOLOv8 model
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

        # COCO class names (YOLOv8 default)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        # Colors for different classes (BGR format)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

    def detect_objects(self, image_path: str) -> Dict:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary containing detection results
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)

            # Extract detection information
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())

                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                            'class': class_name,  # Added for compatibility with GPT-4o analysis
                            'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                            'area': float((x2 - x1) * (y2 - y1))
                        }

                        detections.append(detection)

            return {
                'image_path': image_path,
                'image_shape': image.shape,
                'detections': detections,
                'detection_count': len(detections)
            }

        except Exception as e:
            logger.error(f"Error detecting objects in {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'detections': [],
                'detection_count': 0
            }

    def annotate_frame(self, image_path: str, detections: List[Dict]) -> np.ndarray:
        """
        Annotate frame with bounding boxes and labels
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
        
        Returns:
            Annotated image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        annotated_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']

            # Get color for this class
            color = tuple(map(int, self.colors[class_id % len(self.colors)]))

            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"

            # Calculate label background size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw label background
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1) - label_height - 10),
                (int(x1) + label_width, int(y1)),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated_image

    def process_frame_batch(self, frame_paths: List[str], output_dir: str) -> List[Dict]:
        """
        Process multiple frames and save annotated versions
        
        Args:
            frame_paths: List of paths to frame images
            output_dir: Directory to save annotated frames
        
        Returns:
            List of detection results for all frames
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_results = []

        for i, frame_path in enumerate(frame_paths):
            logger.info(f"Processing frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
            
            try:
                # Detect objects
                result = self.detect_objects(frame_path)
                
                if 'error' not in result:
                    if result['detections']:
                        # Annotate frame only if detections found
                        annotated_image = self.annotate_frame(frame_path, result['detections'])

                        # Save annotated frame
                        frame_filename = os.path.basename(frame_path)
                        name, ext = os.path.splitext(frame_filename)
                        annotated_filename = f"{name}_annotated{ext}"
                        annotated_path = os.path.join(output_dir, annotated_filename)

                        cv2.imwrite(annotated_path, annotated_image)
                        result['annotated_frame_path'] = annotated_path

                        logger.info(f"Found {result['detection_count']} objects in {frame_filename}, saved annotated frame")
                    else:
                        logger.info(f"No objects detected in {os.path.basename(frame_path)}")
                        # Don't create annotated frame if no detections
                        result['annotated_frame_path'] = None
                else:
                    logger.error(f"Error processing {os.path.basename(frame_path)}: {result.get('error', 'Unknown error')}")
                    result['annotated_frame_path'] = None

            except Exception as e:
                logger.error(f"Exception processing frame {frame_path}: {e}")
                result = {
                    'image_path': frame_path,
                    'error': str(e),
                    'detections': [],
                    'detection_count': 0,
                    'annotated_frame_path': None
                }

            all_results.append(result)

        return all_results

    def generate_detection_summary(self, results: List[Dict]) -> Dict:
        """
        Generate summary statistics from detection results
        
        Args:
            results: List of detection results
        
        Returns:
            Summary dictionary
        """
        total_frames = len(results)
        frames_with_detections = sum(1 for r in results if r.get('detection_count', 0) > 0)
        total_detections = sum(r.get('detection_count', 0) for r in results)

        # Count detections by class
        class_counts = {}
        confidence_scores = []

        for result in results:
            for detection in result.get('detections', []):
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidence_scores.append(detection['confidence'])

        # Calculate confidence statistics
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        min_confidence = np.min(confidence_scores) if confidence_scores else 0
        max_confidence = np.max(confidence_scores) if confidence_scores else 0

        summary = {
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'detection_rate': frames_with_detections / total_frames if total_frames > 0 else 0,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'class_counts': class_counts,
            'confidence_stats': {
                'average': float(avg_confidence),
                'minimum': float(min_confidence),
                'maximum': float(max_confidence)
            },
            'most_detected_classes': sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

        return summary

    def save_results_json(self, results: List[Dict], summary: Dict, output_path: str):
        """
        Save detection results and summary to JSON file
        
        Args:
            results: List of detection results
            summary: Summary statistics
            output_path: Path to save JSON file
        """
        try:
            output_data = {
                'summary': summary,
                'detailed_results': results
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise
