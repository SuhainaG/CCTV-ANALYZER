# utils/audio_processor.py

import os
import subprocess
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, assemblyai_api_key: str, elevenlabs_api_key: str = None):
        """
        Initialize AudioProcessor with AssemblyAI integration
        ElevenLabs integration is optional and will use REST API if provided
        """
        self.assemblyai_api_key = assemblyai_api_key
        self.assemblyai_base_url = "https://api.assemblyai.com/v2"
        
        # ElevenLabs setup (using REST API instead of SDK)
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_base_url = "https://api.elevenlabs.io/v1"
        
        if elevenlabs_api_key:
            logger.info("ElevenLabs API initialized (REST API)")
        
        # Enhanced suspicious keywords for security analysis
        self.suspicious_keywords = [
            'bomb', 'kill', 'murder', 'fight', 'attack', 'weapon', 'gun', 'knife',
            'hurt', 'violence', 'threat', 'steal', 'rob', 'drugs', 'illegal',
            'hide', 'secret', 'plan', 'escape', 'emergency', 'help', 'danger',
            'police', 'security', 'alarm', 'break', 'destroy', 'vandalism',
            'kidnap', 'ransom', 'explosive', 'poison', 'terror', 'assault',
            'blackmail', 'fraud', 'hack', 'breach', 'infiltrate', 'sabotage'
        ]
        
        # Enhanced emotion keywords for detailed sentiment analysis
        self.emotion_keywords = {
            'extreme_anger': ['furious', 'enraged', 'livid', 'irate', 'seething'],
            'anger': ['angry', 'mad', 'annoyed', 'irritated', 'upset'],
            'extreme_fear': ['terrified', 'horrified', 'petrified', 'panic', 'dread'],
            'fear': ['scared', 'afraid', 'worried', 'nervous', 'anxious'],
            'extreme_sadness': ['devastated', 'heartbroken', 'despairing', 'anguished'],
            'sadness': ['sad', 'depressed', 'unhappy', 'melancholy', 'gloomy'],
            'extreme_joy': ['ecstatic', 'euphoric', 'elated', 'overjoyed', 'thrilled'],
            'joy': ['happy', 'glad', 'cheerful', 'content', 'pleased'],
            'disgust': ['disgusted', 'revolted', 'appalled', 'sickened'],
            'surprise': ['shocked', 'amazed', 'astonished', 'stunned', 'bewildered'],
            'confusion': ['confused', 'puzzled', 'perplexed', 'baffled'],
            'excitement': ['excited', 'enthusiastic', 'eager', 'pumped'],
            'calm': ['calm', 'peaceful', 'serene', 'relaxed', 'tranquil'],
            'stress': ['stressed', 'overwhelmed', 'pressured', 'tense']
        }

    def generate_elevenlabs_speech(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Optional[bytes]:
        """
        Generate speech using ElevenLabs REST API
        """
        if not self.elevenlabs_api_key:
            logger.warning("ElevenLabs API key not configured")
            return None
        
        try:
            url = f"{self.elevenlabs_base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                logger.info("ElevenLabs speech generated successfully")
                return response.content
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating ElevenLabs speech: {e}")
            return None

    def speed_up_video_for_audio(self, video_path: str, output_path: str, speed_factor: float = 1.25) -> bool:
        """
        Speed up video for faster audio processing while maintaining intelligibility
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-filter:a', f'atempo={speed_factor}',
                '-vn',  # No video
                '-y',  # Overwrite
                output_path
            ]
            
            logger.info(f"Speeding up audio by {speed_factor}x for faster processing...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Audio speed increased successfully: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg speed-up error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error speeding up audio: {e}")
            return False

    def extract_audio_from_video(self, video_path: str, output_audio_path: str, speed_up: bool = True) -> bool:
        """
        Extract and optionally speed up audio from video
        """
        try:
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            
            if speed_up:
                # Create temporary speeded up version
                temp_path = output_audio_path.replace('.mp3', '_temp.mp3')
                if self.speed_up_video_for_audio(video_path, temp_path, 1.25):
                    # Convert to final format
                    cmd = [
                        'ffmpeg',
                        '-i', temp_path,
                        '-acodec', 'libmp3lame',
                        '-ab', '128k',  # Lower bitrate for faster processing
                        '-ar', '22050',  # Lower sample rate for faster processing
                        '-y',
                        output_audio_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    if result.returncode == 0:
                        logger.info(f"Optimized audio extracted: {output_audio_path}")
                        return True
            
            # Fallback to normal extraction
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'libmp3lame',
                '-ab', '128k',
                '-ar', '22050',
                '-y',
                output_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully: {output_audio_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False

    def upload_audio_to_assemblyai(self, audio_path: str) -> Optional[str]:
        """Upload audio with progress tracking"""
        try:
            headers = {
                'authorization': self.assemblyai_api_key,
                'content-type': 'application/octet-stream'
            }
            
            file_size = os.path.getsize(audio_path)
            logger.info(f"Uploading audio file ({file_size / (1024*1024):.1f} MB)...")
            
            with open(audio_path, 'rb') as audio_file:
                response = requests.post(
                    f"{self.assemblyai_base_url}/upload",
                    headers=headers,
                    data=audio_file,
                    timeout=300
                )
            
            if response.status_code == 200:
                upload_url = response.json()['upload_url']
                logger.info("Audio uploaded successfully")
                return upload_url
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading audio: {e}")
            return None

    def transcribe_audio_with_assemblyai(self, upload_url: str) -> Optional[Dict]:
        """Enhanced transcription with all advanced features"""
        try:
            headers = {
                'authorization': self.assemblyai_api_key,
                'content-type': 'application/json'
            }
            
            # Enhanced configuration for maximum accuracy and features
            transcription_config = {
                'audio_url': upload_url,
                'speaker_labels': True,
                'auto_chapters': True,
                'sentiment_analysis': True,
                'entity_detection': True,
                'iab_categories': True,
                'language_detection': True,
                'punctuate': True,
                'format_text': True,
                'dual_channel': False,
                'auto_highlights': True,
                'boost_param': 'high',
                'filter_profanity': False,  # We want to detect everything for security
                'redact_pii': False,  # Keep all information for analysis
                'speed_boost': True  # Use speed boost for faster processing
            }
            
            # Submit transcription job
            response = requests.post(
                f"{self.assemblyai_base_url}/transcript",
                headers=headers,
                json=transcription_config,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Transcription submission failed: {response.status_code} - {response.text}")
                return None
            
            transcript_id = response.json()['id']
            logger.info(f"Enhanced transcription job submitted: {transcript_id}")
            
            # Poll for completion with shorter intervals for speed
            max_wait_time = 1800  # 30 minutes
            poll_interval = 5  # 5 seconds for faster feedback
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                status_response = requests.get(
                    f"{self.assemblyai_base_url}/transcript/{transcript_id}",
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code != 200:
                    logger.error(f"Status check failed: {status_response.status_code}")
                    return None
                
                status_data = status_response.json()
                status = status_data['status']
                
                logger.info(f"Transcription status: {status} ({elapsed_time}s elapsed)")
                
                if status == 'completed':
                    logger.info("Enhanced transcription completed successfully")
                    return status_data
                elif status == 'error':
                    logger.error(f"Transcription failed: {status_data.get('error', 'Unknown error')}")
                    return None
                
                time.sleep(poll_interval)
                elapsed_time += poll_interval
            
            logger.error("Transcription timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error during enhanced transcription: {e}")
            return None

    def analyze_frame_audio_correlation(self, frame_timestamp: float, audio_segments: List[Dict]) -> Dict:
        """Analyze audio context for a specific frame timestamp"""
        try:
            frame_audio_context = {
                'timestamp': frame_timestamp,
                'relevant_segments': [],
                'dominant_emotion': 'neutral',
                'sentiment_score': 0,
                'suspicious_content': False,
                'speaker_info': {},
                'transcription': ''
            }
            
            # Find audio segments within ±2 seconds of frame
            time_window = 2.0
            relevant_segments = []
            
            for segment in audio_segments:
                segment_start = segment.get('start_time', 0)
                segment_end = segment.get('end_time', 0)
                
                if (segment_start <= frame_timestamp + time_window and 
                    segment_end >= frame_timestamp - time_window):
                    relevant_segments.append(segment)
            
            if relevant_segments:
                # Combine transcriptions
                transcriptions = [seg.get('text', '') for seg in relevant_segments]
                combined_text = ' '.join(transcriptions)
                frame_audio_context['transcription'] = combined_text
                
                # Analyze emotions and sentiment
                emotions = []
                sentiment_scores = []
                suspicious_flags = []
                
                for segment in relevant_segments:
                    emotion = segment.get('emotion', 'neutral')
                    sentiment = segment.get('sentiment', 'neutral')
                    suspicious_keywords = segment.get('suspicious_keywords', [])
                    
                    emotions.append(emotion)
                    
                    # Convert sentiment to score
                    if sentiment == 'positive':
                        sentiment_scores.append(1)
                    elif sentiment == 'negative':
                        sentiment_scores.append(-1)
                    else:
                        sentiment_scores.append(0)
                    
                    if suspicious_keywords:
                        suspicious_flags.append(True)
                        frame_audio_context['suspicious_content'] = True
                
                # Determine dominant emotion
                if emotions:
                    emotion_counts = {}
                    for emotion in emotions:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    frame_audio_context['dominant_emotion'] = max(emotion_counts, key=emotion_counts.get)
                
                # Calculate average sentiment
                if sentiment_scores:
                    frame_audio_context['sentiment_score'] = sum(sentiment_scores) / len(sentiment_scores)
                
                # Speaker information
                speakers = set(seg.get('speaker', 'Unknown') for seg in relevant_segments)
                frame_audio_context['speaker_info'] = {
                    'speakers': list(speakers),
                    'speaker_count': len(speakers)
                }
                
                frame_audio_context['relevant_segments'] = relevant_segments
            
            return frame_audio_context
            
        except Exception as e:
            logger.error(f"Error analyzing frame-audio correlation: {e}")
            return {
                'timestamp': frame_timestamp,
                'error': str(e),
                'relevant_segments': [],
                'dominant_emotion': 'neutral',
                'sentiment_score': 0,
                'suspicious_content': False
            }

    def detailed_emotion_analysis(self, text: str) -> Dict:
        """Perform detailed emotion analysis with intensity levels"""
        try:
            text_lower = text.lower()
            emotion_scores = {}
            
            # Analyze each emotion category
            for emotion_category, keywords in self.emotion_keywords.items():
                score = 0
                detected_keywords = []
                
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
                        detected_keywords.append(keyword)
                
                if score > 0:
                    emotion_scores[emotion_category] = {
                        'score': score,
                        'keywords': detected_keywords,
                        'intensity': 'high' if 'extreme_' in emotion_category else 'medium'
                    }
            
            # Determine primary emotion
            primary_emotion = 'neutral'
            max_score = 0
            
            if emotion_scores:
                for emotion, data in emotion_scores.items():
                    if data['score'] > max_score:
                        max_score = data['score']
                        primary_emotion = emotion
            
            return {
                'primary_emotion': primary_emotion,
                'emotion_scores': emotion_scores,
                'emotional_intensity': self._calculate_emotional_intensity(emotion_scores),
                'is_distressed': any('extreme_' in emotion for emotion in emotion_scores.keys())
            }
            
        except Exception as e:
            logger.error(f"Error in detailed emotion analysis: {e}")
            return {
                'primary_emotion': 'neutral',
                'emotion_scores': {},
                'emotional_intensity': 'low',
                'is_distressed': False
            }

    def _calculate_emotional_intensity(self, emotion_scores: Dict) -> str:
        """Calculate overall emotional intensity"""
        if not emotion_scores:
            return 'low'
        
        total_score = sum(data['score'] for data in emotion_scores.values())
        extreme_emotions = sum(1 for emotion in emotion_scores.keys() if 'extreme_' in emotion)
        
        if extreme_emotions > 0 or total_score > 5:
            return 'high'
        elif total_score > 2:
            return 'medium'
        else:
            return 'low'

    def process_transcript_segments_enhanced(self, transcript_data: Dict) -> List[Dict]:
        """Enhanced transcript processing with detailed emotion analysis"""
        try:
            processed_segments = []
            utterances = transcript_data.get('utterances', [])
            
            for utterance in utterances:
                start_time = utterance.get('start', 0) / 1000
                end_time = utterance.get('end', 0) / 1000
                text = utterance.get('text', '')
                
                # Enhanced emotion analysis
                emotion_analysis = self.detailed_emotion_analysis(text)
                
                # Suspicious keyword detection
                suspicious_keywords = []
                text_lower = text.lower()
                for keyword in self.suspicious_keywords:
                    if keyword.lower() in text_lower:
                        suspicious_keywords.append(keyword)
                
                segment = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'speaker': utterance.get('speaker', 'Unknown'),
                    'text': text,
                    'confidence': utterance.get('confidence', 0),
                    'word_count': len(text.split()),
                    'timestamp_formatted': f"{int(start_time // 60):02d}:{int(start_time % 60):02d}",
                    
                    # Enhanced emotion data
                    'emotion': emotion_analysis['primary_emotion'],
                    'emotional_intensity': emotion_analysis['emotional_intensity'],
                    'is_distressed': emotion_analysis['is_distressed'],
                    'emotion_scores': emotion_analysis['emotion_scores'],
                    
                    # Security data
                    'suspicious_keywords': suspicious_keywords,
                    'is_suspicious': len(suspicious_keywords) > 0,
                    
                    # Additional metadata
                    'segment_type': self._classify_segment_type(text, emotion_analysis),
                    'attention_required': len(suspicious_keywords) > 0 or emotion_analysis['is_distressed']
                }
                
                processed_segments.append(segment)
            
            return processed_segments
            
        except Exception as e:
            logger.error(f"Error in enhanced segment processing: {e}")
            return []

    def _classify_segment_type(self, text: str, emotion_analysis: Dict) -> str:
        """Classify the type of audio segment"""
        text_lower = text.lower()
        
        if emotion_analysis['is_distressed']:
            return 'distress'
        elif any(word in text_lower for word in ['help', 'emergency', 'call', 'police']):
            return 'emergency'
        elif any(word in text_lower for word in self.suspicious_keywords):
            return 'suspicious'
        elif emotion_analysis['primary_emotion'] in ['extreme_anger', 'anger']:
            return 'aggressive'
        elif emotion_analysis['primary_emotion'] in ['extreme_fear', 'fear']:
            return 'fearful'
        elif len(text.split()) < 3:
            return 'brief'
        else:
            return 'normal'

    def analyze_transcript_for_suspicious_content(self, transcript_data: Dict) -> Dict:
        """Enhanced suspicious content analysis"""
        try:
            suspicious_analysis = {
                'suspicious_keywords_found': [],
                'suspicious_segments': [],
                'risk_score': 0,
                'sentiment_concerns': [],
                'speaker_analysis': {},
                'threat_level': 'LOW',
                'security_flags': []
            }
            
            # Get transcript and utterances
            full_text = transcript_data.get('text', '').lower()
            utterances = transcript_data.get('utterances', [])
            
            # Enhanced keyword detection
            found_keywords = []
            high_risk_keywords = ['bomb', 'kill', 'murder', 'weapon', 'attack', 'terror']
            
            for keyword in self.suspicious_keywords:
                if keyword.lower() in full_text:
                    found_keywords.append(keyword)
                    if keyword in high_risk_keywords:
                        suspicious_analysis['security_flags'].append(f"HIGH_RISK_KEYWORD: {keyword}")
            
            suspicious_analysis['suspicious_keywords_found'] = found_keywords
            
            # Analyze each utterance
            suspicious_segments = []
            speaker_stats = {}
            
            for utterance in utterances:
                speaker = utterance.get('speaker', 'Unknown')
                text = utterance.get('text', '').lower()
                start_time = utterance.get('start', 0) / 1000
                
                # Track speaker stats
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        'total_utterances': 0,
                        'suspicious_utterances': 0,
                        'high_risk_utterances': 0,
                        'total_words': 0,
                        'risk_score': 0
                    }
                
                speaker_stats[speaker]['total_utterances'] += 1
                speaker_stats[speaker]['total_words'] += len(text.split())
                
                # Check for suspicious content
                utterance_keywords = []
                high_risk_count = 0
                
                for keyword in self.suspicious_keywords:
                    if keyword.lower() in text:
                        utterance_keywords.append(keyword)
                        if keyword in high_risk_keywords:
                            high_risk_count += 1
                
                if utterance_keywords:
                    speaker_stats[speaker]['suspicious_utterances'] += 1
                    if high_risk_count > 0:
                        speaker_stats[speaker]['high_risk_utterances'] += 1
                    
                    speaker_stats[speaker]['risk_score'] += len(utterance_keywords) + (high_risk_count * 5)
                    
                    suspicious_segments.append({
                        'start_time': start_time,
                        'end_time': utterance.get('end', 0) / 1000,
                        'speaker': speaker,
                        'text': utterance.get('text', ''),
                        'keywords': utterance_keywords,
                        'high_risk_keywords': high_risk_count,
                        'confidence': utterance.get('confidence', 0)
                    })
            
            suspicious_analysis['suspicious_segments'] = suspicious_segments
            suspicious_analysis['speaker_analysis'] = speaker_stats
            
            # Calculate enhanced risk score
            risk_score = 0
            risk_score += len(found_keywords) * 10
            risk_score += len(suspicious_segments) * 5
            risk_score += sum(1 for kw in found_keywords if kw in high_risk_keywords) * 20
            
            # Determine threat level
            if risk_score > 80:
                suspicious_analysis['threat_level'] = 'CRITICAL'
                suspicious_analysis['security_flags'].append('CRITICAL_THREAT_DETECTED')
            elif risk_score > 50:
                suspicious_analysis['threat_level'] = 'HIGH'
                suspicious_analysis['security_flags'].append('HIGH_THREAT_DETECTED')
            elif risk_score > 20:
                suspicious_analysis['threat_level'] = 'MEDIUM'
            else:
                suspicious_analysis['threat_level'] = 'LOW'
            
            suspicious_analysis['risk_score'] = risk_score
            
            return suspicious_analysis
            
        except Exception as e:
            logger.error(f"Error in enhanced suspicious content analysis: {e}")
            return {
                'error': str(e),
                'suspicious_keywords_found': [],
                'suspicious_segments': [],
                'risk_score': 0,
                'threat_level': 'UNKNOWN'
            }

    def process_video_audio_enhanced(self, video_path: str, output_dir: str, 
                                   frame_timestamps: List[float] = None) -> Optional[Dict]:
        """Complete enhanced audio processing pipeline with frame correlation"""
        try:
            if not self.assemblyai_api_key:
                logger.warning("AssemblyAI API key not configured")
                return None
            
            logger.info(f"Starting enhanced audio processing for: {video_path}")
            
            # Step 1: Extract optimized audio
            audio_filename = f"optimized_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            audio_path = os.path.join(output_dir, audio_filename)
            
            if not self.extract_audio_from_video(video_path, audio_path, speed_up=True):
                logger.error("Failed to extract audio")
                return None
            
            # Step 2: Upload to AssemblyAI
            upload_url = self.upload_audio_to_assemblyai(audio_path)
            if not upload_url:
                logger.error("Failed to upload audio")
                return None
            
            # Step 3: Enhanced transcription
            transcript_data = self.transcribe_audio_with_assemblyai(upload_url)
            if not transcript_data:
                logger.error("Failed to transcribe audio")
                return None
            
            # Step 4: Enhanced analysis
            suspicious_analysis = self.analyze_transcript_for_suspicious_content(transcript_data)
            processed_segments = self.process_transcript_segments_enhanced(transcript_data)
            
            # Step 5: Frame-audio correlation analysis
            frame_correlations = []
            if frame_timestamps:
                logger.info("Analyzing frame-audio correlations...")
                for timestamp in frame_timestamps:
                    correlation = self.analyze_frame_audio_correlation(timestamp, processed_segments)
                    frame_correlations.append(correlation)
            
            # Step 6: Generate enhanced summary
            enhanced_summary = self.generate_enhanced_audio_summary(
                transcript_data, suspicious_analysis, frame_correlations
            )
            
            # Step 7: Compile enhanced results
            enhanced_results = {
                'processing_timestamp': datetime.now().isoformat(),
                'source_video': video_path,
                'extracted_audio_path': audio_path,
                'basic_info': enhanced_summary.get('basic_info', {}),
                'security_analysis': enhanced_summary.get('security_analysis', {}),
                'emotional_analysis': enhanced_summary.get('emotional_analysis', {}),
                'frame_correlations': frame_correlations,
                'transcript_segments': processed_segments,
                'suspicious_analysis': suspicious_analysis,
                'recommendations': enhanced_summary.get('recommendations', []),
                'full_transcript': transcript_data.get('text', ''),
                'enhanced_summary': enhanced_summary
            }
            
            # Step 8: Generate ElevenLabs summary if available
            if self.elevenlabs_api_key:
                try:
                    summary_text = self.create_audio_summary_text(enhanced_results)
                    audio_data = self.generate_elevenlabs_speech(summary_text)
                    if audio_data:
                        summary_audio_path = os.path.join(output_dir, 'audio_summary.mp3')
                        with open(summary_audio_path, 'wb') as f:
                            f.write(audio_data)
                        enhanced_results['elevenlabs_summary_audio'] = summary_audio_path
                except Exception as e:
                    logger.error(f"ElevenLabs summary generation failed: {e}")
            
            # Save enhanced results
            results_path = os.path.join(output_dir, 'enhanced_audio_analysis.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Enhanced audio processing complete: {results_path}")
            logger.info(f"Risk Level: {enhanced_summary.get('security_analysis', {}).get('risk_level', 'UNKNOWN')}")
            logger.info(f"Emotional State: {enhanced_summary.get('emotional_analysis', {}).get('emotional_stability', 'unknown')}")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in enhanced audio processing: {e}")
            return None

    def generate_enhanced_audio_summary(self, transcript_data: Dict, suspicious_analysis: Dict, 
                                      frame_correlations: List[Dict] = None) -> Dict:
        """Generate comprehensive audio analysis summary with frame correlations"""
        try:
            summary = {
                'basic_info': {},
                'security_analysis': {},
                'emotional_analysis': {},
                'frame_correlations': {},
                'recommendations': []
            }
            
            # Basic information
            duration = transcript_data.get('audio_duration', 0) / 1000
            language = transcript_data.get('language_code', 'unknown')
            
            summary['basic_info'] = {
                'duration_seconds': duration,
                'duration_formatted': f"{int(duration // 60):02d}:{int(duration % 60):02d}",
                'language': language,
                'confidence': transcript_data.get('confidence', 0),
                'total_segments': len(transcript_data.get('utterances', [])),
                'speaker_count': len(set(utt.get('speaker', 'Unknown') 
                                      for utt in transcript_data.get('utterances', [])))
            }
            
            # Security analysis
            risk_score = suspicious_analysis.get('risk_score', 0)
            suspicious_keywords = suspicious_analysis.get('suspicious_keywords_found', [])
            
            summary['security_analysis'] = {
                'risk_level': 'HIGH' if risk_score > 50 else 'MEDIUM' if risk_score > 20 else 'LOW',
                'risk_score': risk_score,
                'suspicious_keywords': suspicious_keywords,
                'suspicious_segments': len(suspicious_analysis.get('suspicious_segments', [])),
                'threat_indicators': len(suspicious_keywords) > 0
            }
            
            # Emotional analysis
            utterances = transcript_data.get('utterances', [])
            all_emotions = []
            distressed_segments = 0
            
            for utterance in utterances:
                text = utterance.get('text', '')
                emotion_analysis = self.detailed_emotion_analysis(text)
                all_emotions.append(emotion_analysis['primary_emotion'])
                
                if emotion_analysis['is_distressed']:
                    distressed_segments += 1
            
            emotion_distribution = {}
            for emotion in all_emotions:
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            summary['emotional_analysis'] = {
                'dominant_emotion': max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else 'neutral',
                'emotion_distribution': emotion_distribution,
                'distressed_segments': distressed_segments,
                'emotional_stability': 'unstable' if distressed_segments > len(utterances) * 0.3 else 'stable'
            }
            
            # Frame correlations
            if frame_correlations:
                correlation_summary = {
                    'total_frames_analyzed': len(frame_correlations),
                    'frames_with_audio': len([f for f in frame_correlations if f.get('transcription')]),
                    'suspicious_audio_frames': len([f for f in frame_correlations if f.get('suspicious_content')]),
                    'emotional_peaks': []
                }
                
                # Find emotional peaks
                for frame_corr in frame_correlations:
                    if abs(frame_corr.get('sentiment_score', 0)) > 0.5:
                        correlation_summary['emotional_peaks'].append({
                            'timestamp': frame_corr.get('timestamp', 0),
                            'emotion': frame_corr.get('dominant_emotion', 'neutral'),
                            'sentiment_score': frame_corr.get('sentiment_score', 0)
                        })
                
                summary['frame_correlations'] = correlation_summary
            
            # Generate recommendations
            recommendations = []
            
            if risk_score > 30:
                recommendations.append("🚨 HIGH SECURITY ALERT: Review suspicious segments immediately")
            
            if distressed_segments > 3:
                recommendations.append("⚠️ Multiple distressed audio segments detected - investigate for welfare concerns")
            
            if summary['emotional_analysis']['emotional_stability'] == 'unstable':
                recommendations.append("📊 Emotional instability detected - analyze for behavioral patterns")
            
            if len(suspicious_keywords) > 0:
                recommendations.append(f"🔍 Suspicious keywords detected: {', '.join(suspicious_keywords[:3])}")
            
            summary['recommendations'] = recommendations
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating enhanced audio summary: {e}")
            return {
                'error': str(e),
                'basic_info': {},
                'security_analysis': {},
                'emotional_analysis': {},
                'recommendations': []
            }

    def create_audio_summary_text(self, results: Dict) -> str:
        """Create a summary text for TTS"""
        try:
            basic_info = results.get('basic_info', {})
            security = results.get('security_analysis', {})
            emotional = results.get('emotional_analysis', {})
            
            summary_parts = [
                f"Audio Analysis Summary for {basic_info.get('duration_formatted', 'unknown duration')} of content.",
                f"Security Risk Level: {security.get('risk_level', 'unknown')}",
                f"Dominant Emotional State: {emotional.get('dominant_emotion', 'neutral')}",
                f"Speakers Detected: {basic_info.get('speaker_count', 0)}",
            ]
            
            if security.get('suspicious_keywords'):
                summary_parts.append(f"Suspicious keywords detected: {', '.join(security['suspicious_keywords'][:3])}")
            
            if results.get('recommendations'):
                summary_parts.append("Recommendations: " + "; ".join(results['recommendations'][:2]))
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating summary text: {e}")
            return "Audio analysis completed with enhanced processing."
