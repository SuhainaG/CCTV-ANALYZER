# utils/accessibility_manager.py

import pyttsx3
import threading
import queue
import time
import re
from typing import Optional, Dict, List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccessibilityManager:
    def __init__(self):
        """
        Initialize AccessibilityManager with comprehensive TTS capabilities
        """
        self.tts_engine = None
        self.tts_queue = queue.Queue()
        self.is_speaking = False
        self.tts_thread = None
        self.shutdown_flag = False
        self.initialize_tts()
        
        # TTS settings optimized for comprehensive content
        self.default_rate = 160  # Slightly slower for complex content
        self.default_volume = 0.9
        self.default_voice = None
        
        # Content-specific rates
        self.rates = {
            'summary': 140,      # Slower for summaries
            'narrative': 150,    # Medium for stories  
            'details': 160,      # Normal for details
            'alerts': 180        # Faster for alerts
        }
        
        # Start TTS worker thread
        self.start_tts_worker()

    def initialize_tts(self):
        """Initialize the TTS engine with comprehensive voice setup"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Set default properties
            self.tts_engine.setProperty('rate', self.default_rate)
            self.tts_engine.setProperty('volume', self.default_volume)
            
            # Get available voices and select best one
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice for better accessibility
                selected_voice = voices[0]  # Default fallback
                
                for voice in voices:
                    voice_name = voice.name.lower()
                    if any(keyword in voice_name for keyword in ['zira', 'hazel', 'susan', 'female']):
                        selected_voice = voice
                        break
                    elif 'microsoft' in voice_name and 'david' not in voice_name:
                        selected_voice = voice
                
                self.tts_engine.setProperty('voice', selected_voice.id)
                self.default_voice = selected_voice.id
                logger.info(f"Selected TTS voice: {selected_voice.name}")
            
            logger.info("Comprehensive TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None

    def start_tts_worker(self):
        """Start the TTS worker thread"""
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

    def _tts_worker(self):
        """Worker thread for processing TTS queue with priority handling"""
        while not self.shutdown_flag:
            try:
                tts_request = self.tts_queue.get(timeout=1)
                
                if tts_request is None:
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
        Synchronously speak text with content-aware settings
        """
        try:
            if not self.tts_engine or self.shutdown_flag:
                logger.error("TTS engine not available")
                return
            
            self.is_speaking = True
            
            # Apply content-aware options
            content_type = options.get('content_type', 'details')
            rate = options.get('rate', self.rates.get(content_type, self.default_rate))
            volume = options.get('volume', self.default_volume)
            voice = options.get('voice', self.default_voice)
            
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', volume)
            if voice:
                self.tts_engine.setProperty('voice', voice)
            
            # Clean and prepare text
            cleaned_text = self.clean_text_for_comprehensive_speech(text, content_type)
            
            logger.info(f"Speaking {content_type} content: {cleaned_text[:100]}...")
            
            # Speak with appropriate pacing
            self.tts_engine.say(cleaned_text)
            self.tts_engine.runAndWait()
            
            logger.info("Finished speaking comprehensive content")
            
        except Exception as e:
            logger.error(f"Error speaking comprehensive text: {e}")
        finally:
            self.is_speaking = False

    def speak_text(self, text: str, options: Optional[Dict] = None) -> bool:
        """
        Queue text for speech with content-aware processing
        """
        try:
            if not self.tts_engine or self.shutdown_flag:
                logger.error("TTS engine not available")
                return False
            
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS")
                return False
            
            # Determine content type if not specified
            if not options:
                options = {}
            
            if 'content_type' not in options:
                options['content_type'] = self._detect_content_type(text)
            
            tts_request = {
                'text': text,
                'options': options
            }
            
            self.tts_queue.put(tts_request)
            logger.info(f"Queued {options['content_type']} content for speech: {len(text)} characters")
            return True
            
        except Exception as e:
            logger.error(f"Error queuing comprehensive text for speech: {e}")
            return False

    def _detect_content_type(self, text: str) -> str:
        """
        Detect content type for appropriate TTS settings
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['summary', 'analysis complete', 'overview']):
            return 'summary'
        elif any(word in text_lower for word in ['story', 'narrative', 'customer', 'activity']):
            return 'narrative'
        elif any(word in text_lower for word in ['alert', 'risk', 'suspicious', 'emergency']):
            return 'alerts'
        else:
            return 'details'

    def clean_text_for_comprehensive_speech(self, text: str, content_type: str = 'details') -> str:
        """
        Clean and prepare text for comprehensive speech synthesis
        """
        try:
            # Remove or replace problematic characters
            cleaned = text.replace('\n', '. ')
            cleaned = cleaned.replace('\t', ' ')
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Enhanced abbreviation replacements for comprehensive content
            replacements = {
                'GPT-4o': 'GPT-4 omni',
                'GPT-4': 'GPT-4',
                'YOLO': 'YOLO object detection',
                'CCTV': 'C C T V',
                'AI': 'artificial intelligence',
                'API': 'A P I',
                'JSON': 'JSON',
                'HTTP': 'H T T P',
                'URL': 'U R L',
                'ID': 'I D',
                'TTS': 'text to speech',
                'vs': 'versus',
                '&': 'and',
                '@': 'at',
                '%': 'percent',
                '#': 'number',
                '$': 'dollars',
                'fps': 'frames per second',
                'HD': 'high definition',
                '4K': '4K resolution',
                'MB': 'megabytes',
                'GB': 'gigabytes',
                'mins': 'minutes',
                'secs': 'seconds'
            }
            
            for abbrev, full_form in replacements.items():
                cleaned = cleaned.replace(abbrev, full_form)
            
            # Content-type specific processing
            if content_type == 'summary':
                # Add pauses for better comprehension
                cleaned = cleaned.replace('. ', '. ... ')
                cleaned = cleaned.replace('Analysis', 'Analysis ... ')
            elif content_type == 'narrative':
                # Add narrative flow indicators
                cleaned = cleaned.replace('[', 'At timestamp ')
                cleaned = cleaned.replace(']', ', ')
            elif content_type == 'alerts':
                # Emphasize alert content
                cleaned = cleaned.replace('HIGH', 'HIGH PRIORITY')
                cleaned = cleaned.replace('RISK', 'RISK DETECTED')
            
            # Remove excessive punctuation
            cleaned = cleaned.replace('...', '. pause. ')
            cleaned = cleaned.replace('!!', '!')
            cleaned = cleaned.replace('??', '?')
            
            # Ensure proper sentence endings
            if cleaned and not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
            
            return cleaned.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning comprehensive text for speech: {e}")
            return text

    def speak_comprehensive_summary(self, summary_data: Dict) -> bool:
        """
        Speak comprehensive video analysis summary
        """
        try:
            summary_text = self._create_comprehensive_summary_speech(summary_data)
            return self.speak_text(summary_text, {'content_type': 'summary'})
        except Exception as e:
            logger.error(f"Error speaking comprehensive summary: {e}")
            return False

    def _create_comprehensive_summary_speech(self, summary_data: Dict) -> str:
        """
        Create speech-optimized comprehensive summary
        """
        try:
            speech_parts = []
            
            # Executive summary
            if summary_data.get('executive_summary'):
                speech_parts.append(f"Executive Summary: {summary_data['executive_summary']}")
            
            # Story analysis
            story_analysis = summary_data.get('story_analysis', {})
            if story_analysis:
                story_text = f"Story Analysis: {story_analysis.get('total_story_moments', 0)} significant moments identified. "
                story_text += f"{story_analysis.get('customer_interactions', 0)} customer interactions analyzed. "
                speech_parts.append(story_text)
            
            # Risk summary
            risk_summary = summary_data.get('risk_summary', {})
            if risk_summary.get('total_incidents', 0) > 0:
                risk_text = f"Security Analysis: {risk_summary['total_incidents']} security incidents detected. "
                risk_text += f"Overall risk level is {risk_summary.get('overall_risk_level', 'unknown')}. "
                speech_parts.append(risk_text)
            
            # Audio summary
            audio_summary = summary_data.get('audio_summary', {})
            if audio_summary:
                audio_text = f"Audio Analysis: {audio_summary.get('speaker_count', 0)} speakers detected. "
                audio_text += f"Emotional stability is {audio_summary.get('emotional_stability', 'unknown')}. "
                speech_parts.append(audio_text)
            
            # Recommendations
            recommendations = summary_data.get('recommendations', [])
            if recommendations:
                speech_parts.append(f"Key Recommendations: {'. '.join(recommendations[:3])}")
            
            return ' ... '.join(speech_parts)
            
        except Exception as e:
            logger.error(f"Error creating comprehensive summary speech: {e}")
            return "Comprehensive video analysis completed successfully."

    def announce_processing_status(self, step: str, progress: int, message: str) -> bool:
        """
        Announce processing status with comprehensive context
        """
        try:
            # Create context-aware announcement
            if progress == 100:
                announcement = f"Analysis complete! {message}"
                content_type = 'summary'
            elif 'risk' in step.lower() or 'suspicious' in message.lower():
                announcement = f"Security analysis: {message}"
                content_type = 'alerts'
            elif progress % 25 == 0:  # Major milestones
                announcement = f"{step} is {progress} percent complete. {message}"
                content_type = 'summary'
            else:
                return True  # Skip minor updates to avoid overwhelming
            
            return self.speak_text(announcement, {'content_type': content_type})
            
        except Exception as e:
            logger.error(f"Error announcing comprehensive processing status: {e}")
            return False

    def speak_frame_analysis(self, frame_data: Dict) -> bool:
        """
        Speak detailed frame analysis
        """
        try:
            frame_text = self._create_frame_analysis_speech(frame_data)
            return self.speak_text(frame_text, {'content_type': 'narrative'})
        except Exception as e:
            logger.error(f"Error speaking frame analysis: {e}")
            return False

    def _create_frame_analysis_speech(self, frame_data: Dict) -> str:
        """
        Create speech-optimized frame analysis
        """
        try:
            analysis = frame_data.get('analysis', {})
            
            speech_parts = [
                f"Frame analysis for {frame_data.get('filename', 'unknown frame')}"
            ]
            
            # Timestamp
            if frame_data.get('timestamp'):
                timestamp = frame_data['timestamp']
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                speech_parts.append(f"at timestamp {minutes} minutes {seconds} seconds")
            
            # Risk and sentiment
            risk_level = analysis.get('risk_level', 'unknown')
            sentiment = analysis.get('sentiment_analysis', 'neutral')
            speech_parts.append(f"Risk level is {risk_level}. Sentiment is {sentiment}")
            
            # Scene summary
            if analysis.get('scene_summary'):
                speech_parts.append(f"Scene description: {analysis['scene_summary']}")
            
            # Story narrative
            if analysis.get('story_narrative'):
                speech_parts.append(f"Story context: {analysis['story_narrative']}")
            
            # Behavioral analysis
            if analysis.get('behavioral_analysis'):
                speech_parts.append(f"Behavioral insights: {analysis['behavioral_analysis']}")
            
            # Suspicious indicators
            if analysis.get('suspicious_indicators') and analysis['suspicious_indicators'] not in ['None identified', 'Manual review required']:
                speech_parts.append(f"Security concern: {analysis['suspicious_indicators']}")
            
            # Audio context
            if frame_data.get('has_audio') and analysis.get('audio_context', {}).get('transcription'):
                speech_parts.append(f"Audio context: {analysis['audio_context']['transcription']}")
            
            return '. '.join(speech_parts) + '.'
            
        except Exception as e:
            logger.error(f"Error creating frame analysis speech: {e}")
            return "Frame analysis details are not available."

    def stop_speaking(self):
        """Stop current speech"""
        try:
            if self.tts_engine and self.is_speaking:
                self.tts_engine.stop()
                logger.info("Comprehensive speech stopped")
        except Exception as e:
            logger.error(f"Error stopping comprehensive speech: {e}")

    def clear_speech_queue(self):
        """Clear all queued speech requests"""
        try:
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except queue.Empty:
                    break
            logger.info("Comprehensive speech queue cleared")
        except Exception as e:
            logger.error(f"Error clearing comprehensive speech queue: {e}")

    def shutdown(self):
        """Shutdown the accessibility manager"""
        try:
            self.shutdown_flag = True
            self.stop_speaking()
            self.clear_speech_queue()
            self.tts_queue.put(None)
            
            if self.tts_thread and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=5)
            
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
                self.tts_engine = None
            
            logger.info("Comprehensive accessibility manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during comprehensive accessibility manager shutdown: {e}")

    def __del__(self):
        try:
            self.shutdown()
        except:
            pass
