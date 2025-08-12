# utils/accessibility_manager.py
import pyttsx3
import threading
import queue
import time
from typing import Optional, Dict, List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccessibilityManager:
    def __init__(self):
        """
        Initialize AccessibilityManager with text-to-speech capabilities
        """
        self.tts_engine = None
        self.tts_queue = queue.Queue()
        self.is_speaking = False
        self.tts_thread = None
        self.initialize_tts()
        
        # TTS settings
        self.default_rate = 180  # Words per minute
        self.default_volume = 0.8  # Volume level (0.0 to 1.0)
        self.default_voice = None  # Will be set to first available voice
      
        # Start TTS worker thread
        self.start_tts_worker()
    
    def initialize_tts(self):
        """Initialize the TTS engine with error handling"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Set default properties
            self.tts_engine.setProperty('rate', self.default_rate)
            self.tts_engine.setProperty('volume', self.default_volume)
            
            # Get available voices and set a default
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available, otherwise use first voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        self.default_voice = voice.id
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
                    self.default_voice = voices[0].id
            
            logger.info("TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
    
    def start_tts_worker(self):
        """Start the TTS worker thread"""
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
    
    def _tts_worker(self):
        """Worker thread for processing TTS queue"""
        while True:
            try:
                # Get next TTS request from queue (blocking)
                tts_request = self.tts_queue.get(timeout=1)
                
                if tts_request is None:  # Shutdown signal
                    break
                
                text = tts_request.get('text', '')
                options = tts_request.get('options', {})
                
                if text and self.tts_engine:
                    self._speak_text_sync(text, options)
                
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS worker: {e}")
    
    def _speak_text_sync(self, text: str, options: Dict):
        """
        Synchronously speak text with given options
        
        Args:
            text: Text to speak
            options: TTS options (rate, volume, voice)
        """
        try:
            if not self.tts_engine:
                logger.error("TTS engine not available")
                return
            
            self.is_speaking = True
            
            # Apply options
            rate = options.get('rate', self.default_rate)
            volume = options.get('volume', self.default_volume)
            voice = options.get('voice', self.default_voice)
            
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', volume)
            if voice:
                self.tts_engine.setProperty('voice', voice)
            
            # Clean text for better speech
            cleaned_text = self.clean_text_for_speech(text)
            
            logger.info(f"Speaking text: {cleaned_text[:100]}...")
            
            # Speak the text
            self.tts_engine.say(cleaned_text)
            self.tts_engine.runAndWait()
            
            logger.info("Finished speaking")
            
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
        finally:
            self.is_speaking = False
    
    def speak_text(self, text: str, options: Optional[Dict] = None) -> bool:
        """
        Queue text for speech (non-blocking)
        
        Args:
            text: Text to speak
            options: Optional TTS settings
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            if not self.tts_engine:
                logger.error("TTS engine not available")
                return False
            
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS")
                return False
            
            tts_request = {
                'text': text,
                'options': options or {}
            }
            
            # Add to queue
            self.tts_queue.put(tts_request)
            logger.info(f"Text queued for speech: {len(text)} characters")
            
            return True
            
        except Exception as e:
            logger.error(f"Error queuing text for speech: {e}")
            return False
    
    def speak_text_sync(self, text: str, options: Optional[Dict] = None) -> bool:
        """
        Speak text synchronously (blocking)
        
        Args:
            text: Text to speak
            options: Optional TTS settings
            
        Returns:
            True if spoken successfully, False otherwise
        """
        try:
            if not self.tts_engine:
                return False
            
            self._speak_text_sync(text, options or {})
            return True
            
        except Exception as e:
            logger.error(f"Error in synchronous speech: {e}")
            return False
    
    def stop_speaking(self):
        """Stop current speech"""
        try:
            if self.tts_engine and self.is_speaking:
                self.tts_engine.stop()
                logger.info("Speech stopped")
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
    
    def clear_speech_queue(self):
        """Clear all queued speech requests"""
        try:
            while not self.tts_queue.empty():
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            logger.info("Speech queue cleared")
        except Exception as e:
            logger.error(f"Error clearing speech queue: {e}")
    
    def clean_text_for_speech(self, text: str) -> str:
        """
        Clean and prepare text for better speech synthesis
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text optimized for speech
        """
        try:
            # Remove or replace problematic characters
            cleaned = text.replace('\n', '. ')  # Replace newlines with pauses
            cleaned = cleaned.replace('\t', ' ')  # Replace tabs with spaces
            cleaned = cleaned.replace('  ', ' ')  # Remove double spaces
            
            # Replace common abbreviations with full words for better pronunciation
            replacements = {
                'GPT-4o': 'GPT-4 omni',
                'YOLO': 'YOLO',
                'CCTV': 'C C T V',
                'AI': 'A I',
                'API': 'A P I',
                'JSON': 'JSON',
                'HTTP': 'H T T P',
                'URL': 'U R L',
                'ID': 'I D',
                'GPS': 'G P S',
                'RGB': 'R G B',
                'FPS': 'F P S',
                'MP4': 'M P 4',
                'TTS': 'text to speech',
                'vs': 'versus',
                '&': 'and',
                '@': 'at',
                '%': 'percent',
                '#': 'number',
                '$': 'dollars'
            }
            
            for abbrev, full_form in replacements.items():
                cleaned = cleaned.replace(abbrev, full_form)
            
            # Remove excessive punctuation
            cleaned = cleaned.replace('...', '.')
            cleaned = cleaned.replace('!!', '!')
            cleaned = cleaned.replace('??', '?')
            
            # Ensure sentences end properly
            if cleaned and not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
            
            return cleaned.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text for speech: {e}")
            return text  # Return original text if cleaning fails
    
    def get_available_voices(self) -> List[Dict]:
        """
        Get list of available TTS voices
        
        Returns:
            List of voice information dictionaries
        """
        try:
            if not self.tts_engine:
                return []
            
            voices = self.tts_engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'age': getattr(voice, 'age', None),
                    'gender': getattr(voice, 'gender', None),
                    'languages': getattr(voice, 'languages', [])
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def set_voice_settings(self, rate: Optional[int] = None, 
                          volume: Optional[float] = None, 
                          voice_id: Optional[str] = None):
        """
        Update TTS voice settings
        
        Args:
            rate: Speaking rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice_id: Voice ID to use
        """
        try:
            if not self.tts_engine:
                return
            
            if rate is not None:
                self.tts_engine.setProperty('rate', max(50, min(400, rate)))
                self.default_rate = rate
            
            if volume is not None:
                self.tts_engine.setProperty('volume', max(0.0, min(1.0, volume)))
                self.default_volume = volume
            
            if voice_id is not None:
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if voice.id == voice_id:
                        self.tts_engine.setProperty('voice', voice_id)
                        self.default_voice = voice_id
                        break
            
            logger.info(f"TTS settings updated - Rate: {self.default_rate}, Volume: {self.default_volume}")
            
        except Exception as e:
            logger.error(f"Error setting voice settings: {e}")
    
    def create_accessibility_summary(self, frame_analysis: Dict, audio_context: Optional[Dict] = None) -> str:
        """
        Create accessibility-friendly summary for screen readers and TTS
        
        Args:
            frame_analysis: Visual analysis results
            audio_context: Optional audio context
            
        Returns:
            Accessibility-friendly summary text
        """
        try:
            summary_parts = []
            
            # Frame information
            gpt_analysis = frame_analysis.get('gpt4o_analysis', {})
            
            # Risk level announcement
            risk_level = gpt_analysis.get('risk_level', 'UNKNOWN')
            if risk_level == 'HIGH':
                summary_parts.append("High risk alert.")
            elif risk_level == 'MODERATE':
                summary_parts.append("Moderate risk detected.")
            else:
                summary_parts.append("Low risk scene.")
            
            # Scene description
            scene_summary = gpt_analysis.get('scene_summary', '')
            if scene_summary:
                summary_parts.append(f"Scene description: {scene_summary}")
            
            # Objects detected
            detections = frame_analysis.get('detections', [])
            if detections:
                objects = [d.get('class_name', 'unknown') for d in detections]
                unique_objects = list(set(objects))
                if len(unique_objects) == 1:
                    summary_parts.append(f"One {unique_objects[0]} detected.")
                else:
                    summary_parts.append(f"Objects detected: {', '.join(unique_objects)}")
            else:
                summary_parts.append("No objects detected in this frame.")
            
            # Audio context if available
            if audio_context:
                speaker = audio_context.get('speaker', '')
                text = audio_context.get('text', '')
                sentiment = audio_context.get('sentiment', 'neutral')
                
                if text:
                    if speaker and speaker != 'Unknown':
                        summary_parts.append(f"Audio from {speaker}: {text}")
                    else:
                        summary_parts.append(f"Audio transcript: {text}")
                    
                    if sentiment in ['negative', 'very_negative']:
                        summary_parts.append("Negative sentiment detected in audio.")
                
                suspicious_keywords = audio_context.get('suspicious_keywords', [])
                if suspicious_keywords:
                    summary_parts.append(f"Suspicious keywords detected: {', '.join(suspicious_keywords)}")
            
            # Actions and recommendations
            likely_actions = gpt_analysis.get('likely_actions', '')
            if likely_actions:
                summary_parts.append(f"Likely actions: {likely_actions}")
            
            return ' '.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating accessibility summary: {e}")
            return "Error creating accessibility summary."
    
    def narrate_frame_analysis(self, frame_analysis: Dict, audio_context: Optional[Dict] = None) -> bool:
        """
        Narrate frame analysis for accessibility
        
        Args:
            frame_analysis: Visual analysis results
            audio_context: Optional audio context
            
        Returns:
            True if narration was queued successfully
        """
        try:
            accessibility_text = self.create_accessibility_summary(frame_analysis, audio_context)
            return self.speak_text(accessibility_text)
            
        except Exception as e:
            logger.error(f"Error narrating frame analysis: {e}")
            return False
    
    def create_navigation_announcement(self, current_frame: int, total_frames: int, 
                                    has_detections: bool, risk_level: str) -> str:
        """
        Create navigation announcement for accessibility
        
        Args:
            current_frame: Current frame number
            total_frames: Total number of frames
            has_detections: Whether frame has object detections
            risk_level: Risk level of current frame
            
        Returns:
            Navigation announcement text
        """
        try:
            announcement_parts = []
            
            # Frame position
            announcement_parts.append(f"Frame {current_frame} of {total_frames}.")
            
            # Detection status
            if has_detections:
                announcement_parts.append("Objects detected.")
            else:
                announcement_parts.append("No objects detected.")
            
            # Risk level
            if risk_level == 'HIGH':
                announcement_parts.append("High risk.")
            elif risk_level == 'MODERATE':
                announcement_parts.append("Moderate risk.")
            else:
                announcement_parts.append("Low risk.")
            
            return ' '.join(announcement_parts)
            
        except Exception as e:
            logger.error(f"Error creating navigation announcement: {e}")
            return f"Frame {current_frame} of {total_frames}."
    
    def announce_processing_status(self, step: str, progress: int, message: str) -> bool:
        """
        Announce processing status for accessibility
        
        Args:
            step: Current processing step
            progress: Progress percentage
            message: Status message
            
        Returns:
            True if announcement was queued successfully
        """
        try:
            announcement = f"{step}. {progress} percent complete. {message}"
            cleaned_announcement = self.clean_text_for_speech(announcement)
            
            # Use higher rate for status announcements to avoid being too slow
            options = {'rate': 200}
            
            return self.speak_text(cleaned_announcement, options)
            
        except Exception as e:
            logger.error(f"Error announcing processing status: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the accessibility manager and cleanup resources"""
        try:
            # Stop current speech
            self.stop_speaking()
            
            # Clear queue
            self.clear_speech_queue()
            
            # Signal worker thread to stop
            self.tts_queue.put(None)
            
            # Wait for worker thread to finish
            if self.tts_thread and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=5)
            
            # Cleanup TTS engine
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
                self.tts_engine = None
            
            logger.info("Accessibility manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during accessibility manager shutdown: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.shutdown()
        except:
            pass