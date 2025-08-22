from flask import Flask, render_template, request, jsonify, send_file, url_for, Response
import os
import json
import shutil
from werkzeug.utils import secure_filename
import threading
import time
from datetime import datetime
import base64
import requests
from PIL import Image
import io
import traceback
import mimetypes

# Import custom modules
from utils.frame_extractor import FrameExtractor
from utils.object_detector import ObjectDetector
from utils.audio_processor import AudioProcessor
from utils.accessibility_manager import AccessibilityManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cctv-analyzer-secret-key-2025'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'extracted_frames'
ANNOTATED_FOLDER = 'annotated_frames'
RESULTS_FOLDER = 'results'
AUDIO_FOLDER = 'audio_processing'

# Create directories
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER, RESULTS_FOLDER, AUDIO_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    
# Create temp directory for TTS
os.makedirs('temp', exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY', '')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Processing status
processing_status = {
    'is_processing': False,
    'current_step': '',
    'progress': 0,
    'message': '',
    'results': None,
    'error': None,
    'audio_results': None,
    'detailed_progress': {},
    'start_time': None,
    'estimated_completion': None,
    'timeline_events': []
}

# Initialize components
frame_extractor = FrameExtractor(similarity_threshold=0.85, min_frame_interval=20)
object_detector = ObjectDetector(confidence_threshold=0.25)
accessibility_manager = AccessibilityManager()

# Initialize audio processor
audio_processor = None
if ASSEMBLYAI_API_KEY:
    audio_processor = AudioProcessor(ASSEMBLYAI_API_KEY, ELEVENLABS_API_KEY)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'webm', 'm4v'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_timeline_event(timestamp, event_type, description, risk_level='LOW', sentiment='NEUTRAL'):
    """Add event to timeline - FIXED to handle single timestamp"""
    global processing_status
    
    # Ensure timestamp is a number
    if isinstance(timestamp, list):
        timestamp = timestamp[0] if timestamp else 0
    elif timestamp is None:
        timestamp = 0
    
    try:
        event = {
            'timestamp': float(timestamp),
            'formatted_time': f"{int(float(timestamp) // 60)}:{int(float(timestamp) % 60):02d}",
            'event_type': event_type,
            'description': description,
            'risk_level': risk_level,
            'sentiment': sentiment,
            'datetime': datetime.now().isoformat()
        }
        
        processing_status['timeline_events'].append(event)
    except Exception as e:
        print(f"Warning: Could not add timeline event: {e}")

def update_processing_status(step, progress, message, detailed_info=None):
    """Enhanced status updates with timeline tracking"""
    global processing_status
    
    processing_status['current_step'] = step
    processing_status['progress'] = progress
    processing_status['message'] = message
    
    # Add to timeline safely
    try:
        current_time = time.time() - (processing_status.get('start_time', time.time()))
        add_timeline_event(
            current_time,
            'PROCESSING',
            f"{step}: {message}",
            risk_level='LOW',
            sentiment='NEUTRAL'
        )
    except Exception as e:
        print(f"Warning: Timeline update failed: {e}")
    
    if detailed_info:
        processing_status['detailed_progress'][step] = detailed_info
    
    # Calculate ETA
    if processing_status['start_time']:
        elapsed = time.time() - processing_status['start_time']
        if progress > 0:
            total_estimated = (elapsed / progress) * 100
            remaining = total_estimated - elapsed
            processing_status['estimated_completion'] = f"{int(remaining // 60)}:{int(remaining % 60):02d}"
    
    print(f"📊 {step}: {progress}% - {message}")

def analyze_frame_with_gpt4o_enhanced(image_path, detection_data, audio_context=None, frame_timestamp=0):
    """Enhanced GPT-4o analysis with comprehensive prompts"""
    
    if not OPENAI_API_KEY:
        return create_fallback_analysis("GPT-4o API key not configured")
    
    try:
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return create_fallback_analysis("Image encoding failed")
        
        # Create comprehensive prompt
        prompt = create_comprehensive_gpt4o_prompt(detection_data, audio_context, frame_timestamp)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "User-Agent": "CCTV-Analyzer/3.0"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert CCTV security analyst with advanced training in threat assessment, behavioral analysis, customer behavior analysis, and comprehensive video storytelling. Provide detailed, actionable insights and complete narratives."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.2
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            parsed_analysis = parse_comprehensive_gpt4o_response(analysis_text)
            
            # Add metadata
            parsed_analysis['api_metadata'] = {
                'model': 'gpt-4o',
                'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                'analysis_timestamp': datetime.now().isoformat(),
                'frame_timestamp': frame_timestamp
            }
            
            return parsed_analysis
        else:
            return create_fallback_analysis(f"GPT-4o API error: {response.status_code}")
            
    except Exception as e:
        print(f"GPT-4o analysis error: {e}")
        return create_fallback_analysis(f"Analysis error: {str(e)}")

def create_comprehensive_gpt4o_prompt(detection_data, audio_context, frame_timestamp):
    """Create comprehensive prompt for detailed analysis"""
    
    detections = detection_data.get('detections', [])
    detection_summary = f"FRAME TIMESTAMP: {frame_timestamp:.1f}s\n"
    
    if detections:
        detection_summary += f"DETECTED OBJECTS ({len(detections)} total):\n"
        for det in detections:
            detection_summary += f"- {det['class_name']} (confidence: {det['confidence']:.2f})\n"
    else:
        detection_summary += "No objects detected in this frame.\n"
    
    audio_section = ""
    if audio_context and audio_context.get('transcription'):
        audio_section = f"""
🎙️ SYNCHRONIZED AUDIO CONTEXT:
- Transcription: "{audio_context.get('transcription', '')}"
- Emotion: {audio_context.get('dominant_emotion', 'neutral')}
- Sentiment: {audio_context.get('sentiment_score', 0)}
- Suspicious Content: {'YES' if audio_context.get('suspicious_content') else 'NO'}
"""
    
    prompt = f"""
🎬 COMPREHENSIVE CCTV VIDEO ANALYSIS REQUEST

{detection_summary}
{audio_section}

📋 PROVIDE DETAILED ANALYSIS FOR ALL SECTIONS:

**SCENE SUMMARY**: Comprehensive 3-4 sentence description of what's happening in this frame.

**STORY NARRATIVE**: Detailed story of this moment - if retail: customer interactions, purchases, quantities, behavior. If residential: activities, people interactions. If story/movie: plot development, character actions, emotions.

**BEHAVIORAL ANALYSIS**: Detailed analysis of human behavior, body language, interactions, intentions.

**CUSTOMER/ACTIVITY ANALYSIS**: If retail/business: count customers, analyze shopping behavior, item interactions, purchase likelihood. If other setting: describe activities in detail.

**RISK ASSESSMENT**: Assign risk level (HIGH/MODERATE/LOW) with detailed justification.

**SENTIMENT ANALYSIS**: Determine emotional tone (POSITIVE/NEGATIVE/NEUTRAL) with reasoning.

**SUSPICIOUS INDICATORS**: List any suspicious activities, concerning behaviors, or security threats.

**ENVIRONMENTAL CONTEXT**: Lighting, setting, time implications, visibility factors.

**DETAILED INSIGHTS**: Additional observations about objects, spatial relationships, sequence of events.

**TIMELINE SIGNIFICANCE**: How this frame fits into the broader video story/sequence.

Provide rich, detailed analysis suitable for comprehensive video understanding and security assessment.
"""
    
    return prompt

def parse_comprehensive_gpt4o_response(response_text):
    """Parse comprehensive GPT-4o response"""
    
    sections = {
        'scene_summary': '',
        'story_narrative': '',
        'behavioral_analysis': '',
        'customer_activity_analysis': '',
        'risk_level': 'LOW',
        'risk_assessment': '',
        'sentiment_analysis': 'NEUTRAL',
        'sentiment_reasoning': '',
        'suspicious_indicators': '',
        'environmental_context': '',
        'detailed_insights': '',
        'timeline_significance': ''
    }
    
    # Parse sections from response
    lines = response_text.split('\n')
    current_section = None
    
    section_mappings = {
        'SCENE SUMMARY': 'scene_summary',
        'STORY NARRATIVE': 'story_narrative',
        'BEHAVIORAL ANALYSIS': 'behavioral_analysis',
        'CUSTOMER/ACTIVITY ANALYSIS': 'customer_activity_analysis',
        'RISK ASSESSMENT': 'risk_assessment',
        'SENTIMENT ANALYSIS': 'sentiment_analysis',
        'SUSPICIOUS INDICATORS': 'suspicious_indicators',
        'ENVIRONMENTAL CONTEXT': 'environmental_context',
        'DETAILED INSIGHTS': 'detailed_insights',
        'TIMELINE SIGNIFICANCE': 'timeline_significance'
    }
    
    for line in lines:
        line = line.strip()
        
        # Check for section headers
        for header, section_key in section_mappings.items():
            if header in line.upper() and ('**' in line or ':' in line):
                current_section = section_key
                continue
        
        # Extract risk level
        if 'HIGH' in line.upper() and ('risk' in line.lower() or current_section == 'risk_assessment'):
            sections['risk_level'] = 'HIGH'
        elif 'MODERATE' in line.upper() and ('risk' in line.lower() or current_section == 'risk_assessment'):
            sections['risk_level'] = 'MODERATE'
        
        # Extract sentiment
        if 'POSITIVE' in line.upper() and ('sentiment' in line.lower() or current_section == 'sentiment_analysis'):
            sections['sentiment_analysis'] = 'POSITIVE'
        elif 'NEGATIVE' in line.upper() and ('sentiment' in line.lower() or current_section == 'sentiment_analysis'):
            sections['sentiment_analysis'] = 'NEGATIVE'
        
        # Add content to current section
        if current_section and line and not line.startswith('**') and not line.startswith('#'):
            if sections[current_section]:
                sections[current_section] += ' ' + line
            else:
                sections[current_section] = line
    
    # Fallback parsing
    if not any(sections.values()):
        sections['scene_summary'] = response_text[:300] + '...' if len(response_text) > 300 else response_text
        sections['story_narrative'] = response_text
    
    # Clean empty sections
    for key in sections:
        if isinstance(sections[key], str) and not sections[key].strip():
            sections[key] = 'Analysis not available for this section'
    
    return sections

def create_fallback_analysis(error_message):
    """Create fallback analysis when GPT-4o fails"""
    return {
        'scene_summary': f'Analysis unavailable: {error_message}',
        'story_narrative': 'Detailed narrative unavailable due to analysis failure',
        'behavioral_analysis': 'Behavioral analysis unavailable',
        'customer_activity_analysis': 'Activity analysis unavailable',
        'risk_level': 'UNKNOWN',
        'risk_assessment': 'Unable to assess risk',
        'sentiment_analysis': 'NEUTRAL',
        'sentiment_reasoning': 'Sentiment analysis unavailable',
        'suspicious_indicators': 'Manual review required',
        'environmental_context': 'Context analysis unavailable',
        'detailed_insights': 'Additional insights unavailable',
        'timeline_significance': 'Timeline context unavailable',
        'api_metadata': {
            'error': error_message,
            'fallback_used': True,
            'analysis_timestamp': datetime.now().isoformat()
        }
    }

def encode_image_to_base64(image_path):
    """Convert image to base64 with optimization"""
    try:
        with Image.open(image_path) as img:
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def process_video_async_enhanced(video_path, session_id):
    """Enhanced video processing with comprehensive analysis - FIXED VERSION"""
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['error'] = None
        processing_status['start_time'] = time.time()
        processing_status['detailed_progress'] = {}
        processing_status['timeline_events'] = []
        
        # Create session directories
        session_frames_dir = os.path.join(FRAMES_FOLDER, session_id)
        session_annotated_dir = os.path.join(ANNOTATED_FOLDER, session_id)
        session_results_dir = os.path.join(RESULTS_FOLDER, session_id)
        session_audio_dir = os.path.join(AUDIO_FOLDER, session_id)
        
        for directory in [session_frames_dir, session_annotated_dir, session_results_dir, session_audio_dir]:
            os.makedirs(directory, exist_ok=True)

        # Get video info
        video_info = frame_extractor.get_video_info(video_path)
        update_processing_status('Video Analysis', 5, f'Analyzing {video_info["duration_formatted"]} video')
        
        add_timeline_event(0, 'VIDEO_START', f'Video analysis started - Duration: {video_info["duration_formatted"]}, Resolution: {video_info["resolution"]}')

        # Step 1: Frame Extraction
        update_processing_status('Frame Extraction', 10, 'Extracting unique frames...')
        frame_paths = frame_extractor.extract_unique_frames(video_path, session_frames_dir)
        
        if not frame_paths:
            raise ValueError("No unique frames extracted from video")
        
        # FIXED: Extract timestamps safely
        frame_timestamps = []
        for frame_path in frame_paths:
            filename = os.path.basename(frame_path)
            if '_t' in filename and 's.jpg' in filename:
                try:
                    timestamp_str = filename.split('_t')[1].split('s.jpg')
                    timestamp = float(timestamp_str)
                    frame_timestamps.append(timestamp)
                except:
                    frame_timestamps.append(0)
            else:
                frame_timestamps.append(0)
        
        update_processing_status('Frame Extraction Complete', 20, f'Extracted {len(frame_paths)} unique frames')
        
        # FIXED: Safe timeline event addition
        try:
            if frame_timestamps and isinstance(frame_timestamps, list) and len(frame_timestamps) > 0:
                first_timestamp = frame_timestamps[0]
            else:
                first_timestamp = 0
            add_timeline_event(first_timestamp, 'FRAME_EXTRACTION', f'{len(frame_paths)} unique frames extracted')
        except Exception as e:
            print(f"Warning: Could not add timeline event - {e}")

        # Step 2: Enhanced Audio Processing
        audio_results = None
        if audio_processor and ASSEMBLYAI_API_KEY:
            update_processing_status('Audio Processing', 25, 'Processing audio with sentiment analysis...')
            try:
                audio_results = audio_processor.process_video_audio_enhanced(
                    video_path, session_audio_dir, frame_timestamps
                )
                processing_status['audio_results'] = audio_results
                
                # Add audio events to timeline
                if audio_results and audio_results.get('transcript_segments'):
                    for segment in audio_results['transcript_segments'][:5]:  # First 5 segments
                        add_timeline_event(
                            segment.get('start_time', 0),
                            'AUDIO_SEGMENT',
                            f"Audio: {segment.get('text', '')[:50]}...",
                            risk_level='HIGH' if segment.get('is_suspicious') else 'LOW',
                            sentiment=segment.get('emotion', 'NEUTRAL').upper()
                        )
                
                update_processing_status('Audio Processing Complete', 40, 'Enhanced audio analysis complete')
            except Exception as e:
                print(f"Audio processing error: {e}")
                update_processing_status('Audio Processing Skipped', 40, 'Audio processing failed, continuing...')

        # Step 3: Object Detection
        update_processing_status('Object Detection', 45, 'Running YOLOv8 object detection...')
        detection_results = object_detector.process_frame_batch(frame_paths, session_annotated_dir)
        
        # Add detection events to timeline
        for i, result in enumerate(detection_results):
            if result.get('detections') and i < len(frame_timestamps):
                timestamp = frame_timestamps[i]
                objects = [d['class_name'] for d in result['detections']]
                add_timeline_event(
                    timestamp,
                    'OBJECT_DETECTION',
                    f"Detected: {', '.join(set(objects))}",
                    risk_level='MODERATE' if 'person' in objects else 'LOW'
                )
        
        update_processing_status('Object Detection Complete', 60, f'Detected objects in {sum(1 for r in detection_results if r["detections"])} frames')

        # Step 4: Comprehensive GPT-4o Analysis
        enhanced_results = []
        frames_with_detections = [r for r in detection_results if r['detections']]
        
        for i, result in enumerate(detection_results):
            progress = 60 + int((i / len(detection_results)) * 30)
            frame_name = os.path.basename(result.get('image_path', ''))
            
            update_processing_status(
                'AI Scene Analysis',
                progress,
                f'Analyzing frame {i+1}/{len(detection_results)}'
            )

            enhanced_result = result.copy()
            
            # Get audio context
            audio_context = None
            frame_timestamp = frame_timestamps[i] if i < len(frame_timestamps) else 0
            
            if audio_results and i < len(frame_timestamps):
                frame_correlations = audio_results.get('frame_correlations', [])
                if i < len(frame_correlations):
                    audio_context = frame_correlations[i]

            # Comprehensive GPT-4o analysis
            if result['detections'] and result.get('annotated_frame_path'):
                annotated_frame_path = result['annotated_frame_path']
                if os.path.exists(annotated_frame_path):
                    gpt_analysis = analyze_frame_with_gpt4o_enhanced(
                        annotated_frame_path, result, audio_context, frame_timestamp
                    )
                    enhanced_result['gpt4o_analysis'] = gpt_analysis
                    enhanced_result['audio_context'] = audio_context
                    enhanced_result['frame_timestamp'] = frame_timestamp
                    
                    # Add analysis events to timeline
                    if gpt_analysis.get('risk_level') in ['HIGH', 'MODERATE']:
                        add_timeline_event(
                            frame_timestamp,
                            'RISK_DETECTED',
                            gpt_analysis.get('scene_summary', 'Risk detected'),
                            risk_level=gpt_analysis.get('risk_level', 'UNKNOWN'),
                            sentiment=gpt_analysis.get('sentiment_analysis', 'NEUTRAL')
                        )
                else:
                    enhanced_result['gpt4o_analysis'] = create_fallback_analysis('Annotated frame not found')
            else:
                enhanced_result['gpt4o_analysis'] = create_fallback_analysis('No objects detected')
                enhanced_result['frame_timestamp'] = frame_timestamp

            enhanced_results.append(enhanced_result)

        update_processing_status('Analysis Complete', 90, 'Generating comprehensive summary...')

        # Step 5: Generate Comprehensive Summary
        comprehensive_summary = generate_video_story_summary(enhanced_results, audio_results, video_info, processing_status['timeline_events'])

        # Step 6: Save Results
        results_json_path = os.path.join(session_results_dir, 'comprehensive_analysis_results.json')
        
        final_results = {
            'session_id': session_id,
            'processing_completed': datetime.now().isoformat(),
            'processing_time_seconds': time.time() - processing_status['start_time'],
            'video_info': video_info,
            'extracted_frames_count': len(frame_paths),
            'detailed_results': enhanced_results,
            'audio_results': audio_results,
            'comprehensive_summary': comprehensive_summary,
            'timeline_events': processing_status['timeline_events'],
            'session_dirs': {
                'frames': session_frames_dir,
                'annotated': session_annotated_dir,
                'results': session_results_dir,
                'audio': session_audio_dir
            },
            'overall_sentiment': calculate_overall_sentiment(enhanced_results, audio_results),
            'story_analysis': comprehensive_summary.get('story_analysis', {})
        }
        
        # Save comprehensive results
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, default=str, ensure_ascii=False)

        add_timeline_event(
            video_info.get('duration_seconds', 0),
            'ANALYSIS_COMPLETE',
            f'Video analysis completed successfully',
            risk_level='LOW',
            sentiment='POSITIVE'
        )

        update_processing_status('Complete', 100, 'Comprehensive analysis complete!')
        processing_status['results'] = final_results
        
        # TTS summary
        try:
            summary_text = comprehensive_summary.get('executive_summary', 'Analysis complete')
            accessibility_manager.speak_text(f"Video analysis complete. {summary_text}")
        except Exception as e:
            print(f"TTS error: {e}")

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"❌ Processing Error: {e}")
        print(traceback.format_exc())
        processing_status['error'] = error_msg
        update_processing_status('Error', 0, error_msg)
    finally:
        processing_status['is_processing'] = False

def generate_video_story_summary(enhanced_results, audio_results, video_info, timeline_events):
    """Generate comprehensive video story summary"""
    
    # Collect all story narratives
    story_parts = []
    customer_activities = []
    risk_incidents = []
    sentiment_analysis = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    for i, result in enumerate(enhanced_results):
        gpt_analysis = result.get('gpt4o_analysis', {})
        timestamp = result.get('frame_timestamp', i * 5)
        
        # Collect story narrative
        narrative = gpt_analysis.get('story_narrative', '')
        if narrative and 'not available' not in narrative.lower():
            story_parts.append({
                'timestamp': timestamp,
                'narrative': narrative,
                'scene_summary': gpt_analysis.get('scene_summary', '')
            })
        
        # Collect customer activities
        customer_analysis = gpt_analysis.get('customer_activity_analysis', '')
        if customer_analysis and 'not available' not in customer_analysis.lower():
            customer_activities.append({
                'timestamp': timestamp,
                'activity': customer_analysis
            })
        
        # Collect risk incidents
        if gpt_analysis.get('risk_level') in ['HIGH', 'MODERATE']:
            risk_incidents.append({
                'timestamp': timestamp,
                'risk_level': gpt_analysis.get('risk_level'),
                'description': gpt_analysis.get('suspicious_indicators', gpt_analysis.get('scene_summary', ''))
            })
        
        # Count sentiment
        sentiment = gpt_analysis.get('sentiment_analysis', 'NEUTRAL')
        sentiment_analysis[sentiment] = sentiment_analysis.get(sentiment, 0) + 1
    
    # Generate executive summary
    duration = video_info.get('duration_formatted', 'unknown duration')
    total_frames = len(enhanced_results)
    
    executive_summary = f"Comprehensive analysis of {duration} video with {total_frames} analyzed frames. "
    
    if story_parts:
        executive_summary += f"Video contains detailed activities spanning {len(story_parts)} significant moments. "
    
    if customer_activities:
        executive_summary += f"Customer/activity analysis reveals {len(customer_activities)} notable interactions. "
    
    if risk_incidents:
        executive_summary += f"Security analysis identified {len(risk_incidents)} risk incidents requiring attention. "
    else:
        executive_summary += "No significant security risks detected. "
    
    # Generate detailed story
    detailed_story = create_detailed_story_narrative(story_parts, customer_activities, video_info)
    
    # Audio summary
    audio_summary = {}
    if audio_results:
        audio_summary = {
            'language': audio_results.get('basic_info', {}).get('language', 'Unknown'),
            'speaker_count': audio_results.get('basic_info', {}).get('speaker_count', 0),
            'emotional_stability': audio_results.get('emotional_analysis', {}).get('emotional_stability', 'stable'),
            'risk_level': audio_results.get('security_analysis', {}).get('risk_level', 'LOW'),
            'key_phrases': audio_results.get('suspicious_keywords', [])[:5]
        }
    
    return {
        'executive_summary': executive_summary,
        'detailed_story': detailed_story,
        'story_analysis': {
            'total_story_moments': len(story_parts),
            'customer_interactions': len(customer_activities),
            'timeline_events': len(timeline_events),
            'sentiment_distribution': sentiment_analysis
        },
        'risk_summary': {
            'total_incidents': len(risk_incidents),
            'risk_incidents': risk_incidents[:10],  # Top 10
            'overall_risk_level': 'HIGH' if any(r['risk_level'] == 'HIGH' for r in risk_incidents) else 'MODERATE' if risk_incidents else 'LOW'
        },
        'audio_summary': audio_summary,
        'recommendations': generate_recommendations(risk_incidents, sentiment_analysis, audio_results)
    }

def create_detailed_story_narrative(story_parts, customer_activities, video_info):
    """Create detailed narrative story"""
    
    if not story_parts and not customer_activities:
        return "No detailed story narrative available from the analysis."
    
    narrative = f"Video Story Analysis ({video_info.get('duration_formatted', 'unknown duration')}):\n\n"
    
    # Combine and sort all events by timestamp
    all_events = []
    
    for story in story_parts:
        all_events.append({
            'timestamp': story['timestamp'],
            'type': 'NARRATIVE',
            'content': story['narrative']
        })
    
    for activity in customer_activities:
        all_events.append({
            'timestamp': activity['timestamp'],
            'type': 'ACTIVITY',
            'content': activity['activity']
        })
    
    # Sort by timestamp
    all_events.sort(key=lambda x: x['timestamp'])
    
    # Create chronological narrative
    for i, event in enumerate(all_events):
        timestamp = event['timestamp']
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        narrative += f"[{minutes:02d}:{seconds:02d}] "
        
        if event['type'] == 'NARRATIVE':
            narrative += f"Scene: {event['content']}\n\n"
        else:
            narrative += f"Activity: {event['content']}\n\n"
    
    return narrative

def calculate_overall_sentiment(enhanced_results, audio_results):
    """Calculate overall video sentiment"""
    
    sentiment_scores = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    # From visual analysis
    for result in enhanced_results:
        gpt_analysis = result.get('gpt4o_analysis', {})
        sentiment = gpt_analysis.get('sentiment_analysis', 'NEUTRAL')
        sentiment_scores[sentiment] = sentiment_scores.get(sentiment, 0) + 1
    
    # From audio analysis
    if audio_results:
        emotional_analysis = audio_results.get('emotional_analysis', {})
        dominant_emotion = emotional_analysis.get('dominant_emotion', 'neutral')
        
        if 'joy' in dominant_emotion or 'happy' in dominant_emotion:
            sentiment_scores['POSITIVE'] += 3
        elif 'sad' in dominant_emotion or 'anger' in dominant_emotion:
            sentiment_scores['NEGATIVE'] += 3
        else:
            sentiment_scores['NEUTRAL'] += 1
    
    # Determine overall sentiment
    overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    confidence = sentiment_scores[overall_sentiment] / sum(sentiment_scores.values()) if sum(sentiment_scores.values()) > 0 else 0
    
    return {
        'overall_sentiment': overall_sentiment,
        'confidence': confidence,
        'distribution': sentiment_scores,
        'description': get_sentiment_description(overall_sentiment, confidence)
    }

def get_sentiment_description(sentiment, confidence):
    """Get sentiment description"""
    confidence_level = "high" if confidence > 0.6 else "moderate" if confidence > 0.4 else "low"
    
    descriptions = {
        'POSITIVE': f"The video shows predominantly positive sentiment with {confidence_level} confidence. Activities and interactions appear upbeat and constructive.",
        'NEGATIVE': f"The video contains predominantly negative sentiment with {confidence_level} confidence. There may be concerning or distressing content.",
        'NEUTRAL': f"The video maintains neutral sentiment with {confidence_level} confidence. Activities appear routine without strong emotional indicators."
    }
    
    return descriptions.get(sentiment, "Sentiment analysis unavailable")

def generate_recommendations(risk_incidents, sentiment_analysis, audio_results):
    """Generate actionable recommendations"""
    
    recommendations = []
    
    # Risk-based recommendations
    high_risk_count = len([r for r in risk_incidents if r['risk_level'] == 'HIGH'])
    
    if high_risk_count > 0:
        recommendations.append(f"🚨 IMMEDIATE ACTION: {high_risk_count} high-risk incidents detected - review immediately")
        recommendations.append("Consider alerting security personnel or law enforcement")
    
    if len(risk_incidents) > 5:
        recommendations.append("Multiple security incidents detected - implement enhanced monitoring")
    
    # Sentiment-based recommendations
    negative_sentiment = sentiment_analysis.get('NEGATIVE', 0)
    total_sentiment = sum(sentiment_analysis.values())
    
    if negative_sentiment > total_sentiment * 0.5:
        recommendations.append("High negative sentiment detected - investigate potential issues or conflicts")
    
    # Audio-based recommendations
    if audio_results:
        audio_risk = audio_results.get('security_analysis', {}).get('risk_level', 'LOW')
        if audio_risk in ['HIGH', 'CRITICAL']:
            recommendations.append("High-risk audio content detected - review audio transcript for potential threats")
        
        emotional_stability = audio_results.get('emotional_analysis', {}).get('emotional_stability', 'stable')
        if emotional_stability == 'unstable':
            recommendations.append("Emotional distress detected in audio - consider welfare check if applicable")
    
    # Default recommendations
    if not recommendations:
        recommendations.append("✅ No immediate security concerns identified")
        recommendations.append("Continue routine monitoring and maintain current security protocols")
        recommendations.append("Regular review of video analysis results recommended")
    
    return recommendations

# Flask Routes
@app.route('/')
def index():
    return render_template('enhanced_index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Another video is currently being processed. Please wait.'}), 400

    # Reset processing status
    processing_status = {
        'is_processing': False,
        'current_step': '',
        'progress': 0,
        'message': '',
        'results': None,
        'error': None,
        'audio_results': None,
        'detailed_progress': {},
        'start_time': None,
        'estimated_completion': None,
        'timeline_events': []
    }

    try:
        video_path = None
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Handle file upload or local path
        if 'video_file' in request.files and request.files['video_file'].filename:
            file = request.files['video_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                video_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(video_path)
            else:
                return jsonify({'error': 'Invalid file type. Supported: MP4, AVI, MOV, MKV, WMV, WEBM, M4V'}), 400
        elif 'local_path' in request.form and request.form['local_path'].strip():
            local_path = request.form['local_path'].strip()
            if os.path.exists(local_path) and allowed_file(local_path):
                video_path = local_path
            else:
                return jsonify({'error': 'Local file not found or invalid file type.'}), 400
        else:
            return jsonify({'error': 'Please provide either a video file or local path.'}), 400

        print(f"🎬 Starting comprehensive video processing: {video_path}")

        # Start processing
        thread = threading.Thread(target=process_video_async_enhanced, args=(video_path, session_id))
        thread.daemon = True
        thread.start()

        return jsonify({
            'message': 'Comprehensive video processing started',
            'session_id': session_id,
            'features_enabled': {
                'gpt4o_analysis': bool(OPENAI_API_KEY),
                'audio_processing': bool(ASSEMBLYAI_API_KEY),
                'elevenlabs_tts': bool(ELEVENLABS_API_KEY),
                'comprehensive_summary': True,
                'timeline_analysis': True,
                'sentiment_analysis': True
            }
        })

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    return jsonify(processing_status)

@app.route('/results')
def get_results():
    if processing_status['results']:
        return jsonify(processing_status['results'])
    else:
        return jsonify({'error': 'No results available'}), 404

@app.route('/annotated_frames/<session_id>')
def get_annotated_frames(session_id):
    """Get annotated frames with comprehensive analysis"""
    session_annotated_dir = os.path.join(ANNOTATED_FOLDER, session_id)
    if not os.path.exists(session_annotated_dir):
        return jsonify({'error': 'Session not found'}), 404

    # Load results
    results_file = os.path.join(RESULTS_FOLDER, session_id, 'comprehensive_analysis_results.json')
    frame_analyses = {}
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for result in data.get('detailed_results', []):
                frame_path = result.get('image_path', '')
                frame_name = os.path.basename(frame_path)
                if 'gpt4o_analysis' in result:
                    analysis = result['gpt4o_analysis']
                    analysis['audio_context'] = result.get('audio_context', {})
                    analysis['detections'] = result.get('detections', [])
                    analysis['frame_timestamp'] = result.get('frame_timestamp', 0)
                    frame_analyses[frame_name] = analysis
        except Exception as e:
            print(f"Error loading frame analyses: {e}")

    # Get annotated images
    image_files = []
    for filename in os.listdir(session_annotated_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            original_frame = filename.replace('_annotated', '').replace('annotated_', '')
            
            analysis = frame_analyses.get(original_frame, {
                'scene_summary': 'Analysis not available',
                'risk_level': 'UNKNOWN',
                'sentiment_analysis': 'NEUTRAL',
                'story_narrative': 'Not available'
            })
            
            image_info = {
                'filename': filename,
                'url': url_for('serve_image', session_id=session_id, filename=filename),
                'analysis': analysis,
                'has_audio': bool(analysis.get('audio_context', {}).get('transcription')),
                'detection_count': len(analysis.get('detections', [])),
                'timestamp': analysis.get('frame_timestamp', 0)
            }
            image_files.append(image_info)

    # Sort by timestamp
    image_files.sort(key=lambda x: x['timestamp'])
    
    return jsonify({
        'images': image_files,
        'total_count': len(image_files),
        'high_risk_count': len([img for img in image_files if img['analysis'].get('risk_level') == 'HIGH']),
        'moderate_risk_count': len([img for img in image_files if img['analysis'].get('risk_level') == 'MODERATE']),
        'with_audio_count': len([img for img in image_files if img['has_audio']]),
        'positive_sentiment_count': len([img for img in image_files if img['analysis'].get('sentiment_analysis') == 'POSITIVE']),
        'negative_sentiment_count': len([img for img in image_files if img['analysis'].get('sentiment_analysis') == 'NEGATIVE'])
    })

@app.route('/images/<session_id>/<filename>')
def serve_image(session_id, filename):
    """Serve annotated frame images"""
    session_annotated_dir = os.path.join(ANNOTATED_FOLDER, session_id)
    file_path = os.path.join(session_annotated_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "Image not found", 404

@app.route('/audio/<session_id>/<filename>')
def serve_audio(session_id, filename):
    """Serve audio files with proper mimetype"""
    session_audio_dir = os.path.join(AUDIO_FOLDER, session_id)
    file_path = os.path.join(session_audio_dir, filename)
    
    if os.path.exists(file_path):
        mimetype = mimetypes.guess_type(file_path)[0] or 'audio/mpeg'
        return send_file(file_path, mimetype=mimetype)
    else:
        return "Audio not found", 404

@app.route('/download_results/<session_id>')
def download_results(session_id):
    """Download comprehensive analysis results"""
    results_file = os.path.join(RESULTS_FOLDER, session_id, 'comprehensive_analysis_results.json')
    
    if os.path.exists(results_file):
        return send_file(results_file, as_attachment=True, 
                        download_name=f'comprehensive_cctv_analysis_{session_id}.json')
    else:
        return "Results file not found", 404

@app.route('/timeline/<session_id>')
def get_timeline(session_id):
    """Get timeline events for session"""
    results_file = os.path.join(RESULTS_FOLDER, session_id, 'comprehensive_analysis_results.json')
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            timeline_events = data.get('timeline_events', [])
            return jsonify({
                'timeline_events': timeline_events,
                'total_events': len(timeline_events)
            })
        except Exception as e:
            return jsonify({'error': f'Error loading timeline: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Session not found'}), 404

# Accessibility endpoints
@app.route('/accessibility/speak', methods=['POST'])
def speak_text():
    """Enhanced TTS with audio streaming"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        options = data.get('options', {})
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Try pyttsx3 first
        success = accessibility_manager.speak_text(text, options)
        
        # Fallback to ElevenLabs REST API with streaming
        if not success and ELEVENLABS_API_KEY and audio_processor:
            try:
                audio_data = audio_processor.generate_elevenlabs_speech(text)
                if audio_data:
                    # Save audio and return URL for browser playback
                    timestamp = int(time.time())
                    audio_filename = f'tts_{timestamp}.mp3'
                    audio_path = os.path.join('temp', audio_filename)
                    
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data)
                    
                    audio_url = url_for('serve_temp_audio', filename=audio_filename)
                    return jsonify({
                        'success': True,
                        'audio_url': audio_url,
                        'method': 'elevenlabs'
                    })
            except Exception as e:
                print(f"ElevenLabs TTS error: {e}")
        
        return jsonify({'success': success, 'method': 'pyttsx3'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/temp_audio/<filename>')
def serve_temp_audio(filename):
    """Serve temporary audio files"""
    file_path = os.path.join('temp', filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/mpeg')
    else:
        return "Audio not found", 404

@app.route('/accessibility/stop', methods=['POST'])
def stop_speaking():
    """Stop current speech"""
    try:
        accessibility_manager.stop_speaking()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Comprehensive CCTV Analyzer v3.0 - FIXED VERSION")
    print("=" * 60)
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("🖼️ Frames folder:", FRAMES_FOLDER)
    print("🎯 Annotated folder:", ANNOTATED_FOLDER)
    print("📊 Results folder:", RESULTS_FOLDER)
    print("🎵 Audio folder:", AUDIO_FOLDER)
    print("=" * 60)
    print("🔧 Feature Status:")
    print(f"🤖 GPT-4o Analysis: {'✅ Enabled' if OPENAI_API_KEY else '❌ Disabled (Set OPENAI_API_KEY)'}")
    print(f"🎙️ Audio Processing: {'✅ Enabled' if ASSEMBLYAI_API_KEY else '❌ Disabled (Set ASSEMBLYAI_API_KEY)'}")
    print(f"🎵 ElevenLabs TTS: {'✅ Enabled' if ELEVENLABS_API_KEY else '❌ Disabled (Set ELEVENLABS_API_KEY)'}")
    print(f"📊 Comprehensive Analysis: ✅ Enabled")
    print(f"⏰ Timeline Feature: ✅ Enabled")
    print(f"😊 Sentiment Analysis: ✅ Enabled")
    print(f"♿ Accessibility: ✅ Enabled")
    print("=" * 60)
    print("🌐 Access the application at: http://localhost:5000")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    finally:
        accessibility_manager.shutdown()
