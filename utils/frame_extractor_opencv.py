import cv2
import numpy as np
import os
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, similarity_threshold: float = 0.85, min_frame_interval: int = 30):
        """
        Initialize FrameExtractor with OpenCV-based frame similarity detection
        
        Args:
            similarity_threshold: Similarity threshold above which frames are considered similar
            min_frame_interval: Minimum number of frames between extractions
        """
        self.similarity_threshold = similarity_threshold
        self.min_frame_interval = min_frame_interval
        
    def calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate similarity between two frames using histogram comparison
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        try:
            # Resize frames to same size for comparison
            h1, w1 = frame1.shape
            h2, w2 = frame2.shape
            
            if (h1, w1) != (h2, w2):
                # Resize to smaller dimensions
                target_h = min(h1, h2)
                target_w = min(w1, w2)
                frame1 = cv2.resize(frame1, (target_w, target_h))
                frame2 = cv2.resize(frame2, (target_w, target_h))
            
            # Method 1: Histogram comparison
            hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
            hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Method 2: Template matching (more accurate but slower)
            result = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF_NORMED)
            template_correlation = np.max(result)
            
            # Method 3: Mean Squared Error (MSE) based similarity
            mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
            max_pixel_value = 255.0
            mse_similarity = 1.0 - (mse / (max_pixel_value ** 2))
            
            # Combine all methods (weighted average)
            combined_similarity = (
                hist_correlation * 0.3 + 
                template_correlation * 0.4 + 
                mse_similarity * 0.3
            )
            
            return max(0.0, min(1.0, combined_similarity))
            
        except Exception as e:
            logger.warning(f"Error calculating frame similarity: {e}")
            return 0.0  # Assume frames are different if calculation fails
    
    def extract_unique_frames(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extract unique frames from video using OpenCV-based similarity comparison
        
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
                
                # Convert to grayscale and resize for faster comparison
                current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_frame_resized = self.resize_frame_for_comparison(current_frame_gray)
                
                # Extract first frame or compare with previous unique frame
                should_extract = False
                
                if previous_frame_gray is None:
                    should_extract = True
                    logger.info("Extracting first frame")
                else:
                    # Calculate similarity between current and previous unique frame
                    try:
                        similarity_score = self.calculate_frame_similarity(previous_frame_gray, current_frame_resized)
                        
                        if similarity_score < self.similarity_threshold:
                            should_extract = True
                            logger.info(f"Frame {frame_count}: Similarity = {similarity_score:.3f} < {self.similarity_threshold} - Extracting")
                        else:
                            logger.debug(f"Frame {frame_count}: Similarity = {similarity_score:.3f} - Skipping")
                    except Exception as e:
                        logger.warning(f"Similarity calculation failed for frame {frame_count}: {e}. Extracting frame.")
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
            # If no frames extracted, try extracting every Nth frame as fallback
            logger.info("Attempting fallback extraction...")
            frame_paths = self.fallback_frame_extraction(video_path, output_dir)
        
        return frame_paths
    
    def fallback_frame_extraction(self, video_path: str, output_dir: str, interval: int = 300) -> List[str]:
        """
        Fallback method: Extract frames at regular intervals
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            interval: Extract every Nth frame
            
        Returns:
            List of paths to extracted frame images
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Fallback extraction: every {interval} frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % interval == 0:
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    frame_filename = f"frame_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success:
                        frame_paths.append(frame_path)
                        extracted_count += 1
                        logger.info(f"Fallback saved frame {extracted_count}: {frame_filename}")
                
                frame_count += 1
                
        except Exception as e:
            logger.error(f"Error during fallback extraction: {e}")
        finally:
            cap.release()
            
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
    
    def resize_frame_for_comparison(self, frame: np.ndarray, target_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
        """
        Resize frame for faster similarity comparison while maintaining aspect ratio
        
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