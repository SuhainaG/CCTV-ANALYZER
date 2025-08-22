import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, similarity_threshold: float = 0.85, min_frame_interval: int = 30):
        """
        Initialize FrameExtractor with SSIM-based unique frame detection
        
        Args:
            similarity_threshold: SSIM threshold below which frames are considered unique
            min_frame_interval: Minimum number of frames between extractions
        """
        self.similarity_threshold = similarity_threshold
        self.min_frame_interval = min_frame_interval

    def extract_unique_frames(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extract unique frames from video using SSIM comparison
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
        
        Returns:
            List of paths to extracted frame images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frame_paths = []
        previous_frame_gray = None
        frame_count = 0
        extracted_count = 0
        last_extracted_frame = -self.min_frame_interval

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Minimum frame interval: {self.min_frame_interval}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if minimum interval not reached
                if frame_count - last_extracted_frame < self.min_frame_interval:
                    frame_count += 1
                    continue

                # Convert to grayscale and resize for faster SSIM comparison
                current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_frame_resized = self.resize_frame_for_comparison(current_frame_gray)

                # Extract first frame or compare with previous unique frame
                should_extract = False
                
                if previous_frame_gray is None:
                    should_extract = True
                    logger.info("Extracting first frame")
                else:
                    # Calculate SSIM between current and previous unique frame
                    try:
                        similarity_score = ssim(previous_frame_gray, current_frame_resized, data_range=255)
                        if similarity_score < self.similarity_threshold:
                            should_extract = True
                            logger.info(f"Frame {frame_count}: SSIM = {similarity_score:.3f} < {self.similarity_threshold} - Extracting")
                        else:
                            logger.debug(f"Frame {frame_count}: SSIM = {similarity_score:.3f} - Skipping")
                    except Exception as e:
                        logger.warning(f"SSIM calculation failed for frame {frame_count}: {e}. Extracting frame.")
                        should_extract = True

                if should_extract:
                    # Save frame
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    frame_filename = f"frame_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)

                    # Save with high quality
                    success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success:
                        frame_paths.append(frame_path)
                        # Update reference frame for comparison
                        previous_frame_gray = current_frame_resized.copy()
                        last_extracted_frame = frame_count
                        extracted_count += 1
                        logger.info(f"Saved frame {extracted_count}: {frame_filename}")
                    else:
                        logger.error(f"Failed to save frame: {frame_path}")

                frame_count += 1

                # Progress logging
                if frame_count % 1000 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Extracted: {extracted_count}")

        except Exception as e:
            logger.error(f"Error during frame extraction: {e}")
            raise
        finally:
            cap.release()

        logger.info(f"Extraction complete: {extracted_count} unique frames from {frame_count} total frames")

        if not frame_paths:
            logger.warning("No frames were extracted. Check video file and parameters.")

        return frame_paths

    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about the video file
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary containing video information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            info = {
                'file_path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'duration_formatted': f"{int(duration // 60):02d}:{int(duration % 60):02d}",
                'resolution': f"{width}x{height}"
            }

            logger.info(f"Video info: {info['resolution']} @ {fps:.2f}fps, duration: {info['duration_formatted']}")
            return info

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
        finally:
            cap.release()

    def resize_frame_for_comparison(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Resize frame for faster SSIM comparison while maintaining aspect ratio
        
        Args:
            frame: Input frame (grayscale)
            target_size: Target size (width, height)
        
        Returns:
            Resized frame
        """
        try:
            height, width = frame.shape[:2]
            target_width, target_height = target_size

            # Calculate scaling factor to maintain aspect ratio
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Ensure minimum size
            if new_width < 64 or new_height < 64:
                new_width = max(64, new_width)
                new_height = max(64, new_height)

            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized

        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame  # Return original if resize fails

    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate if the video file can be opened and read
        
        Args:
            video_path: Path to video file
        
        Returns:
            True if video is valid, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            # Try to read first frame
            ret, frame = cap.read()
            cap.release()

            return ret and frame is not None

        except Exception as e:
            logger.error(f"Error validating video file {video_path}: {e}")
            return False
