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
    def __init__(self, assemblyai_api_key: str):
        """
        Initialize AudioProcessor with AssemblyAI integration
        
        Args:
            assemblyai_api_key: AssemblyAI API key for transcription services
        """
        self.assemblyai_api_key = assemblyai_api_key
        self.assemblyai_base_url = "https://api.assemblyai.com/v2"
        
        # Suspicious keywords to monitor
        self.suspicious_keywords = [
            'bomb', 'kill', 'murder', 'fight', 'attack', 'weapon', 'gun', 'knife',
            'hurt', 'violence', 'threat', 'steal', 'rob', 'drugs', 'illegal',
            'hide', 'secret', 'plan', 'escape', 'emergency', 'help', 'danger',
            'police', 'security', 'alarm', 'break', 'destroy', 'vandalism'
        ]
        
        # Language detection patterns
        self.language_codes = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
    
    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> bool:
        """
        Extract audio from video using FFmpeg
        
        Args:
            video_path: Path to input video file
            output_audio_path: Path for output audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            
            # FFmpeg command to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'libmp3lame',  # MP3 codec
                '-ab', '192k',  # Audio bitrate
                '-ar', '44100',  # Sample rate
                '-y',  # Overwrite output file
                output_audio_path
            ]
            
            logger.info(f"Extracting audio: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully to: {output_audio_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def upload_audio_to_assemblyai(self, audio_path: str) -> Optional[str]:
        """
        Upload audio file to AssemblyAI for processing
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Upload URL if successful, None otherwise
        """
        try:
            headers = {
                'authorization': self.assemblyai_api_key,
                'content-type': 'application/octet-stream'
            }
            
            with open(audio_path, 'rb') as audio_file:
                response = requests.post(
                    f"{self.assemblyai_base_url}/upload",
                    headers=headers,
                    data=audio_file,
                    timeout=300
                )
            
            if response.status_code == 200:
                upload_url = response.json()['upload_url']
                logger.info(f"Audio uploaded successfully: {upload_url}")
                return upload_url
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading audio: {e}")
            return None
    
    def transcribe_audio_with_assemblyai(self, upload_url: str) -> Optional[Dict]:
        """
        Transcribe audio using AssemblyAI with advanced features
        
        Args:
            upload_url: URL of uploaded audio file
            
        Returns:
            Transcription results if successful, None otherwise
        """
        try:
            headers = {
                'authorization': self.assemblyai_api_key,
                'content-type': 'application/json'
            }
            
            # Advanced transcription configuration
            transcription_config = {
                'audio_url': upload_url,
                'speaker_labels': True,  # Speaker diarization
                'auto_chapters': True,   # Automatic chapter detection
                'sentiment_analysis': True,  # Sentiment analysis
                'entity_detection': True,   # Named entity recognition
                'iab_categories': True,     # Content categorization
                'language_detection': True, # Automatic language detection
                'punctuate': True,
                'format_text': True,
                'dual_channel': False,
                'webhook_url': None,
                'auto_highlights': True,
                'boost_param': 'high'  # Enhanced accuracy
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
            logger.info(f"Transcription job submitted: {transcript_id}")
            
            # Poll for completion
            max_wait_time = 1800  # 30 minutes
            poll_interval = 10    # 10 seconds
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
                
                logger.info(f"Transcription status: {status}")
                
                if status == 'completed':
                    logger.info("Transcription completed successfully")
                    return status_data
                elif status == 'error':
                    logger.error(f"Transcription failed: {status_data.get('error', 'Unknown error')}")
                    return None
                
                time.sleep(poll_interval)
                elapsed_time += poll_interval
            
            logger.error("Transcription timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
    
    def analyze_transcript_for_suspicious_content(self, transcript_data: Dict) -> Dict:
        """
        Analyze transcript for suspicious keywords and content
        
        Args:
            transcript_data: AssemblyAI transcript response
            
        Returns:
            Dictionary with suspicious content analysis
        """
        try:
            suspicious_analysis = {
                'suspicious_keywords_found': [],
                'suspicious_segments': [],
                'risk_score': 0,
                'sentiment_concerns': [],
                'speaker_analysis': {}
            }
            
            # Get transcript text and utterances
            full_text = transcript_data.get('text', '').lower()
            utterances = transcript_data.get('utterances', [])
            
            # Check for suspicious keywords in full text
            found_keywords = []
            for keyword in self.suspicious_keywords:
                if keyword.lower() in full_text:
                    found_keywords.append(keyword)
            
            suspicious_analysis['suspicious_keywords_found'] = found_keywords
            
            # Analyze each utterance for suspicious content
            suspicious_segments = []
            sentiment_concerns = []
            speaker_stats = {}
            
            for utterance in utterances:
                speaker = utterance.get('speaker', 'Unknown')
                text = utterance.get('text', '').lower()
                confidence = utterance.get('confidence', 0)
                start_time = utterance.get('start', 0) / 1000  # Convert to seconds
                end_time = utterance.get('end', 0) / 1000
                
                # Track speaker statistics
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        'total_utterances': 0,
                        'suspicious_utterances': 0,
                        'negative_sentiment_count': 0,
                        'total_words': 0
                    }
                
                speaker_stats[speaker]['total_utterances'] += 1
                speaker_stats[speaker]['total_words'] += len(text.split())
                
                # Check for suspicious keywords in this utterance
                utterance_keywords = []
                for keyword in self.suspicious_keywords:
                    if keyword.lower() in text:
                        utterance_keywords.append(keyword)
                
                if utterance_keywords:
                    speaker_stats[speaker]['suspicious_utterances'] += 1
                    suspicious_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'speaker': speaker,
                        'text': utterance.get('text', ''),
                        'keywords': utterance_keywords,
                        'confidence': confidence
                    })
            
            suspicious_analysis['suspicious_segments'] = suspicious_segments
            suspicious_analysis['speaker_analysis'] = speaker_stats
            
            # Process sentiment analysis if available
            if 'sentiment_analysis_results' in transcript_data:
                for sentiment in transcript_data['sentiment_analysis_results']:
                    if sentiment.get('sentiment') in ['NEGATIVE']:
                        start_time = sentiment.get('start', 0) / 1000
                        end_time = sentiment.get('end', 0) / 1000
                        
                        sentiment_concerns.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'sentiment': sentiment.get('sentiment'),
                            'confidence': sentiment.get('confidence', 0),
                            'text': sentiment.get('text', '')
                        })
                        
                        # Find corresponding speaker
                        for utterance in utterances:
                            utt_start = utterance.get('start', 0) / 1000
                            utt_end = utterance.get('end', 0) / 1000
                            if (utt_start <= start_time <= utt_end or 
                                utt_start <= end_time <= utt_end):
                                speaker = utterance.get('speaker', 'Unknown')
                                if speaker in speaker_stats:
                                    speaker_stats[speaker]['negative_sentiment_count'] += 1
                                break
            
            suspicious_analysis['sentiment_concerns'] = sentiment_concerns
            
            # Calculate overall risk score
            risk_score = 0
            risk_score += len(found_keywords) * 10  # 10 points per suspicious keyword
            risk_score += len(suspicious_segments) * 5  # 5 points per suspicious segment
            risk_score += len(sentiment_concerns) * 3   # 3 points per negative sentiment
            
            suspicious_analysis['risk_score'] = risk_score
            
            return suspicious_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing suspicious content: {e}")
            return {
                'error': str(e),
                'suspicious_keywords_found': [],
                'suspicious_segments': [],
                'risk_score': 0
            }
    
    def process_transcript_segments(self, transcript_data: Dict) -> List[Dict]:
        """
        Process transcript into structured segments with metadata
        
        Args:
            transcript_data: Complete AssemblyAI transcript response
            
        Returns:
            List of processed transcript segments
        """
        try:
            processed_segments = []
            utterances = transcript_data.get('utterances', [])
            sentiment_results = transcript_data.get('sentiment_analysis_results', [])
            
            # Create sentiment lookup for quick access
            sentiment_lookup = {}
            for sentiment in sentiment_results:
                start_ms = sentiment.get('start', 0)
                end_ms = sentiment.get('end', 0)
                sentiment_lookup[(start_ms, end_ms)] = {
                    'sentiment': sentiment.get('sentiment', 'NEUTRAL').lower(),
                    'confidence': sentiment.get('confidence', 0)
                }
            
            for utterance in utterances:
                start_time = utterance.get('start', 0) / 1000  # Convert to seconds
                end_time = utterance.get('end', 0) / 1000
                speaker = utterance.get('speaker', 'Unknown')
                text = utterance.get('text', '')
                confidence = utterance.get('confidence', 0)
                
                # Find matching sentiment
                segment_sentiment = 'neutral'
                sentiment_confidence = 0
                
                for (sent_start, sent_end), sent_data in sentiment_lookup.items():
                    sent_start_sec = sent_start / 1000
                    sent_end_sec = sent_end / 1000
                    
                    # Check for overlap
                    if (sent_start_sec <= start_time <= sent_end_sec or 
                        sent_start_sec <= end_time <= sent_end_sec or
                        start_time <= sent_start_sec <= end_time):
                        segment_sentiment = sent_data['sentiment']
                        sentiment_confidence = sent_data['confidence']
                        break
                
                # Check for suspicious keywords in this segment
                suspicious_keywords = []
                text_lower = text.lower()
                for keyword in self.suspicious_keywords:
                    if keyword.lower() in text_lower:
                        suspicious_keywords.append(keyword)
                
                # Determine emotion based on text analysis
                emotion = self.detect_emotion_from_text(text)
                
                segment = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'speaker': speaker,
                    'text': text,
                    'confidence': confidence,
                    'sentiment': segment_sentiment,
                    'sentiment_confidence': sentiment_confidence,
                    'suspicious_keywords': suspicious_keywords,
                    'emotion': emotion,
                    'word_count': len(text.split()),
                    'timestamp_formatted': f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
                }
                
                processed_segments.append(segment)
            
            return processed_segments
            
        except Exception as e:
            logger.error(f"Error processing transcript segments: {e}")
            return []
    
    def detect_emotion_from_text(self, text: str) -> str:
        """
        Simple emotion detection based on keywords and patterns
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected emotion as string
        """
        text_lower = text.lower()
        
        # Emotion keyword mappings
        emotion_keywords = {
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'damn', 'hell', 'stupid'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'nervous', 'panic', 'danger'],
            'sadness': ['sad', 'depressed', 'cry', 'tears', 'hurt', 'pain', 'sorry'],
            'joy': ['happy', 'glad', 'excited', 'great', 'awesome', 'wonderful', 'love'],
            'disgust': ['disgusting', 'gross', 'sick', 'terrible', 'awful', 'horrible'],
            'surprise': ['wow', 'amazing', 'incredible', 'unbelievable', 'shocked']
        }
        
        # Count emotion indicators
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return dominant emotion or neutral
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        return 'neutral'
    
    def generate_audio_summary(self, transcript_data: Dict, suspicious_analysis: Dict) -> str:
        """
        Generate comprehensive audio analysis summary
        
        Args:
            transcript_data: Complete transcript data
            suspicious_analysis: Suspicious content analysis
            
        Returns:
            Formatted summary string
        """
        try:
            summary_parts = []
            
            # Basic information
            duration = transcript_data.get('audio_duration', 0) / 1000  # Convert to seconds
            language = transcript_data.get('language_code', 'unknown')
            language_name = self.language_codes.get(language, language.upper())
            
            summary_parts.append(f"Audio Duration: {int(duration // 60):02d}:{int(duration % 60):02d}")
            summary_parts.append(f"Language: {language_name}")
            summary_parts.append(f"Transcription Confidence: {transcript_data.get('confidence', 0):.2f}")
            
            # Speaker analysis
            utterances = transcript_data.get('utterances', [])
            speakers = set(utt.get('speaker', 'Unknown') for utt in utterances)
            summary_parts.append(f"Speakers Detected: {len(speakers)}")
            
            # Suspicious content summary
            risk_score = suspicious_analysis.get('risk_score', 0)
            keywords_found = suspicious_analysis.get('suspicious_keywords_found', [])
            suspicious_segments = suspicious_analysis.get('suspicious_segments', [])
            
            summary_parts.append(f"Security Risk Score: {risk_score}")
            if keywords_found:
                summary_parts.append(f"Suspicious Keywords: {', '.join(keywords_found)}")
            summary_parts.append(f"Suspicious Segments: {len(suspicious_segments)}")
            
            # Sentiment analysis
            sentiment_results = transcript_data.get('sentiment_analysis_results', [])
            sentiment_counts = {}
            for sentiment in sentiment_results:
                sent_type = sentiment.get('sentiment', 'NEUTRAL')
                sentiment_counts[sent_type] = sentiment_counts.get(sent_type, 0) + 1
            
            if sentiment_counts:
                summary_parts.append("Overall Sentiment Distribution:")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(sentiment_results)) * 100 if sentiment_results else 0
                    summary_parts.append(f"  {sentiment}: {count} segments ({percentage:.1f}%)")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating audio summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def process_video_audio(self, video_path: str, output_dir: str) -> Optional[Dict]:
        """
        Complete audio processing pipeline for video file
        
        Args:
            video_path: Path to input video file
            output_dir: Directory for audio processing outputs
            
        Returns:
            Dictionary with complete audio analysis results
        """
        try:
            if not self.assemblyai_api_key or self.assemblyai_api_key == "your-assemblyai-api-key-here":
                logger.warning("AssemblyAI API key not configured, skipping audio processing")
                return None
            
            logger.info(f"Starting audio processing for: {video_path}")
            
            # Step 1: Extract audio from video
            audio_filename = f"extracted_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            audio_path = os.path.join(output_dir, audio_filename)
            
            if not self.extract_audio_from_video(video_path, audio_path):
                logger.error("Failed to extract audio from video")
                return None
            
            # Step 2: Upload audio to AssemblyAI
            upload_url = self.upload_audio_to_assemblyai(audio_path)
            if not upload_url:
                logger.error("Failed to upload audio to AssemblyAI")
                return None
            
            # Step 3: Transcribe audio with advanced features
            transcript_data = self.transcribe_audio_with_assemblyai(upload_url)
            if not transcript_data:
                logger.error("Failed to transcribe audio")
                return None
            
            # Step 4: Analyze for suspicious content
            suspicious_analysis = self.analyze_transcript_for_suspicious_content(transcript_data)
            
            # Step 5: Process transcript segments
            processed_segments = self.process_transcript_segments(transcript_data)
            
            # Step 6: Generate summary
            audio_summary = self.generate_audio_summary(transcript_data, suspicious_analysis)
            
            # Step 7: Compile results
            audio_results = {
                'processing_timestamp': datetime.now().isoformat(),
                'source_video': video_path,
                'extracted_audio_path': audio_path,
                'language': self.language_codes.get(transcript_data.get('language_code', 'en'), 'English'),
                'language_code': transcript_data.get('language_code', 'en'),
                'audio_duration_seconds': transcript_data.get('audio_duration', 0) / 1000,
                'transcription_confidence': transcript_data.get('confidence', 0),
                'speaker_count': len(set(seg['speaker'] for seg in processed_segments)),
                'total_segments': len(processed_segments),
                'transcript_segments': processed_segments,
                'suspicious_analysis': suspicious_analysis,
                'suspicious_keywords': suspicious_analysis.get('suspicious_keywords_found', []),
                'risk_score': suspicious_analysis.get('risk_score', 0),
                'overall_sentiment': self.determine_overall_sentiment(processed_segments),
                'summary': audio_summary,
                'full_transcript': transcript_data.get('text', ''),
                'raw_assemblyai_response': transcript_data
            }
            
            # Save audio results to JSON
            results_path = os.path.join(output_dir, 'audio_analysis_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                # Create a copy without the raw response for cleaner output
                clean_results = audio_results.copy()
                clean_results.pop('raw_assemblyai_response', None)
                json.dump(clean_results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Audio processing complete. Results saved to: {results_path}")
            logger.info(f"Language: {audio_results['language']}, Risk Score: {audio_results['risk_score']}")
            logger.info(f"Speakers: {audio_results['speaker_count']}, Segments: {audio_results['total_segments']}")
            
            return audio_results
            
        except Exception as e:
            logger.error(f"Error in audio processing pipeline: {e}")
            return None
    
    def determine_overall_sentiment(self, segments: List[Dict]) -> str:
        """
        Determine overall sentiment from all segments
        
        Args:
            segments: List of processed transcript segments
            
        Returns:
            Overall sentiment as string
        """
        try:
            if not segments:
                return 'neutral'
            
            sentiment_counts = {}
            for segment in segments:
                sentiment = segment.get('sentiment', 'neutral')
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # Return dominant sentiment
            if sentiment_counts:
                return max(sentiment_counts, key=sentiment_counts.get)
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error determining overall sentiment: {e}")
            return 'neutral'