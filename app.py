from flask import Flask, render_template, request, jsonify, send_file, url_for
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

# Import our custom modules
from utils.frame_extractor import FrameExtractor
from utils.object_detector import ObjectDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'extracted_frames'
ANNOTATED_FOLDER = 'annotated_frames'
RESULTS_FOLDER = 'results'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# GPT-4o Configuration - Replace with your actual OpenAI API key
OPENAI_API_KEY = "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" # Replace with your actual API key
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Global variables for processing status
processing_status = {
    'is_processing': False,
    'current_step': '',
    'progress': 0,
    'message': '',
    'results': None,
    'error': None
}

# Initialize components
frame_extractor = FrameExtractor(similarity_threshold=0.85, min_frame_interval=30)
object_detector = ObjectDetector()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_processing_status(step, progress, message):
    """Update global processing status"""
    processing_status['current_step'] = step
    processing_status['progress'] = progress
    processing_status['message'] = message
    print(f"Status Update: {step} - {progress}% - {message}")

def encode_image_to_base64(image_path):
    """Convert image to base64 for GPT-4o"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def analyze_frame_with_gpt4o(image_path, detection_data):
    """Analyze frame using GPT-4o with YOLO detection data"""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-actual-openai-api-key-here":
        return {
            'scene_summary': 'GPT-4o analysis not available (API key not configured)',
            'object_positions': 'Please configure OpenAI API key in app.py',
            'likely_actions': 'API key required for scene analysis',
            'detailed_description': 'Configure OpenAI API key to enable detailed scene analysis'
        }
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return {
                'scene_summary': 'Error: Could not process image',
                'object_positions': 'Image processing failed',
                'likely_actions': 'Unable to analyze',
                'detailed_description': 'Image encoding error'
            }
        
        # Prepare detection metadata for GPT-4o
        detection_info = []
        for detection in detection_data.get('detections', []):
            detection_info.append({
                'class': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            })
        
        detection_summary = f"""
        YOLO Detection Results:
        - Total detections: {len(detection_info)}
        - Detected objects: {', '.join([d['class'] for d in detection_info])}
        - Detection details: {json.dumps(detection_info, indent=2)}
        """
        
        # GPT-4o prompt for scene analysis
        prompt = f"""
        You are an expert CCTV footage analyst. Analyze this frame with the provided YOLO detection data.
        
        {detection_summary}
        
        Please provide a comprehensive analysis in the following format:
        
        SCENE SUMMARY: Describe the overall scene, environment, and context in 2-3 sentences.
        
        OBJECT POSITIONS: Detail the spatial relationships and positions of detected objects.
        
        LIKELY ACTIONS: Infer what activities or behaviors are occurring or likely to occur.
        
        DETAILED DESCRIPTION: Provide a thorough narrative description of the scene, focusing on security-relevant observations, human behavior, object interactions, potential risks or anomalies, and environmental context.
        
        Be specific, factual, and security-focused in your analysis.
        """
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            # Parse the structured response
            analysis = parse_gpt4o_response(analysis_text)
            return analysis
        else:
            error_msg = f"GPT-4o API error: {response.status_code}"
            print(f"GPT-4o API Error: {response.status_code} - {response.text}")
            return {
                'scene_summary': error_msg,
                'object_positions': 'API request failed',
                'likely_actions': 'Unable to analyze',
                'detailed_description': f'Error: {response.text}'
            }
            
    except Exception as e:
        error_msg = f"GPT-4o analysis error: {str(e)}"
        print(f"GPT-4o Analysis Error: {e}")
        return {
            'scene_summary': error_msg,
            'object_positions': 'Analysis failed',
            'likely_actions': 'Error occurred',
            'detailed_description': error_msg
        }

def parse_gpt4o_response(response_text):
    """Parse structured GPT-4o response into components"""
    try:
        sections = {
            'scene_summary': '',
            'object_positions': '',
            'likely_actions': '',
            'detailed_description': ''
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'SCENE SUMMARY' in line.upper():
                current_section = 'scene_summary'
                continue
            elif 'OBJECT POSITIONS' in line.upper():
                current_section = 'object_positions'
                continue
            elif 'LIKELY ACTIONS' in line.upper():
                current_section = 'likely_actions'
                continue
            elif 'DETAILED DESCRIPTION' in line.upper():
                current_section = 'detailed_description'
                continue
            elif current_section and line and not line.startswith('#') and not line.startswith('-'):
                if sections[current_section]:
                    sections[current_section] += ' ' + line
                else:
                    sections[current_section] = line
        
        # If sections are empty, try to extract from the full text
        if not any(sections.values()):
            sections['detailed_description'] = response_text
            sections['scene_summary'] = response_text[:200] + '...' if len(response_text) > 200 else response_text
            sections['object_positions'] = 'Analysis available in detailed description'
            sections['likely_actions'] = 'Analysis available in detailed description'
        
        # Clean up empty sections
        for key in sections:
            if not sections[key].strip():
                sections[key] = 'Analysis not available for this section'
        
        return sections
        
    except Exception as e:
        print(f"Error parsing GPT-4o response: {e}")
        return {
            'scene_summary': 'Error parsing analysis',
            'object_positions': 'Parsing failed',
            'likely_actions': 'Unable to parse',
            'detailed_description': response_text or 'No analysis available'
        }

def process_video_async(video_path, session_id):
    """Process video in background thread with GPT-4o analysis"""
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['error'] = None
        
        # Create session-specific directories
        session_frames_dir = os.path.join(FRAMES_FOLDER, session_id)
        session_annotated_dir = os.path.join(ANNOTATED_FOLDER, session_id)
        session_results_dir = os.path.join(RESULTS_FOLDER, session_id)
        
        for directory in [session_frames_dir, session_annotated_dir, session_results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Step 1: Extract unique frames
        update_processing_status('Extracting unique frames', 10, 'Analyzing video and extracting unique frames using SSIM...')
        
        frame_paths = frame_extractor.extract_unique_frames(video_path, session_frames_dir)
        
        if not frame_paths:
            raise ValueError("No unique frames extracted from video")
        
        update_processing_status('Frame extraction complete', 20, f'Extracted {len(frame_paths)} unique frames')
        
        # Step 2: Object detection on extracted frames
        update_processing_status('Detecting objects', 30, 'Running YOLOv8 object detection on extracted frames...')
        
        detection_results = object_detector.process_frame_batch(frame_paths, session_annotated_dir)
        
        update_processing_status('Object detection complete', 50, 'Starting GPT-4o scene analysis...')
        
        # Step 3: GPT-4o Analysis for each frame with detections
        enhanced_results = []
        frames_with_detections = [result for result in detection_results if result['detections']]
        
        print(f"Found {len(frames_with_detections)} frames with detections out of {len(detection_results)} total frames")
        
        for i, result in enumerate(detection_results):
            update_processing_status(
                'Analyzing scenes with GPT-4o', 
                50 + int((i / len(detection_results)) * 40),
                f'Analyzing frame {i+1}/{len(detection_results)} with GPT-4o...'
            )
            
            # Enhanced result starts with detection result
            enhanced_result = result.copy()
            
            # Only analyze frames with detections using GPT-4o
            if result['detections'] and 'annotated_frame_path' in result:
                annotated_frame_path = result['annotated_frame_path']
                if os.path.exists(annotated_frame_path):
                    gpt_analysis = analyze_frame_with_gpt4o(annotated_frame_path, result)
                    enhanced_result['gpt4o_analysis'] = gpt_analysis
                else:
                    enhanced_result['gpt4o_analysis'] = {
                        'scene_summary': 'Annotated frame not found',
                        'object_positions': 'Frame processing error',
                        'likely_actions': 'Unable to analyze',
                        'detailed_description': 'Annotated frame file not accessible'
                    }
            else:
                enhanced_result['gpt4o_analysis'] = {
                    'scene_summary': 'No objects detected in this frame',
                    'object_positions': 'No objects detected',
                    'likely_actions': 'No significant activity detected',
                    'detailed_description': 'This frame contains no detected objects of interest'
                }
            
            enhanced_results.append(enhanced_result)
        
        update_processing_status('GPT-4o analysis complete', 90, 'Generating final summary...')
        
        # Step 4: Generate summary
        summary = object_detector.generate_detection_summary(detection_results)
        
        # Add GPT-4o summary statistics
        summary['gpt4o_analyzed_frames'] = len(frames_with_detections)
        summary['frames_with_scene_analysis'] = len([r for r in enhanced_results if 'gpt4o_analysis' in r])
        
        # Save enhanced results to JSON
        results_json_path = os.path.join(session_results_dir, 'detection_results.json')
        save_enhanced_results_json(enhanced_results, summary, results_json_path)
        
        # Get video info
        video_info = frame_extractor.get_video_info(video_path)
        
        # Prepare final results with GPT-4o analysis
        final_results = {
            'session_id': session_id,
            'video_info': video_info,
            'extracted_frames_count': len(frame_paths),
            'detection_summary': summary,
            'detailed_results': enhanced_results,
            'session_dirs': {
                'frames': session_frames_dir,
                'annotated': session_annotated_dir,
                'results': session_results_dir
            }
        }
        
        update_processing_status(
            'Complete', 
            100, 
            f'Processing complete! Analyzed {len(enhanced_results)} frames with {len(frames_with_detections)} containing detections'
        )
        processing_status['results'] = final_results
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"Processing Error: {e}")
        processing_status['error'] = error_msg
        update_processing_status('Error', 0, error_msg)
    finally:
        processing_status['is_processing'] = False

def save_enhanced_results_json(results, summary, output_path):
    """Save enhanced results with GPT-4o analysis to JSON"""
    try:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
            
    except Exception as e:
        print(f"Error saving enhanced results: {e}")

@app.route('/')
def index():
    return render_template('index.html')

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
        'error': None
    }
    
    try:
        video_path = None
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check if file upload or local path
        if 'video_file' in request.files and request.files['video_file'].filename:
            # File upload
            file = request.files['video_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                video_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(video_path)
            else:
                return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WMV files.'}), 400
                
        elif 'local_path' in request.form and request.form['local_path'].strip():
            # Local path
            local_path = request.form['local_path'].strip()
            if os.path.exists(local_path) and allowed_file(local_path):
                video_path = local_path
            else:
                return jsonify({'error': 'Local file not found or invalid file type.'}), 400
        else:
            return jsonify({'error': 'Please provide either a video file or local path.'}), 400
        
        print(f"Starting video processing for: {video_path}")
        
        # Start processing in background thread
        thread = threading.Thread(target=process_video_async, args=(video_path, session_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Video processing started',
            'session_id': session_id
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
    """Get list of annotated frame images with GPT-4o analysis for display"""
    session_annotated_dir = os.path.join(ANNOTATED_FOLDER, session_id)
    
    if not os.path.exists(session_annotated_dir):
        return jsonify({'error': 'Session not found'}), 404
    
    # Get results with GPT-4o analysis
    results_file = os.path.join(RESULTS_FOLDER, session_id, 'detection_results.json')
    frame_analyses = {}
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                for result in data.get('detailed_results', []):
                    frame_path = result.get('image_path', '')
                    frame_name = os.path.basename(frame_path)
                    if 'gpt4o_analysis' in result:
                        frame_analyses[frame_name] = result['gpt4o_analysis']
        except Exception as e:
            print(f"Error loading frame analyses: {e}")
    
    # Get all image files from annotated directory
    image_files = []
    for filename in os.listdir(session_annotated_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Find corresponding original frame name
            original_frame = filename.replace('_annotated', '').replace('annotated_', '')
            
            image_info = {
                'filename': filename,
                'url': url_for('serve_image', session_id=session_id, filename=filename),
                'gpt4o_analysis': frame_analyses.get(original_frame, {
                    'scene_summary': 'Analysis not available',
                    'object_positions': 'Not analyzed',
                    'likely_actions': 'Not analyzed',
                    'detailed_description': 'GPT-4o analysis not available for this frame'
                })
            }
            image_files.append(image_info)
    
    # Sort by filename
    image_files.sort(key=lambda x: x['filename'])
    
    return jsonify({'images': image_files})

@app.route('/images/<session_id>/<filename>')
def serve_image(session_id, filename):
    """Serve annotated frame images"""
    session_annotated_dir = os.path.join(ANNOTATED_FOLDER, session_id)
    file_path = os.path.join(session_annotated_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "Image not found", 404

@app.route('/download_results/<session_id>')
def download_results(session_id):
    """Download enhanced results JSON file with GPT-4o analysis"""
    results_file = os.path.join(RESULTS_FOLDER, session_id, 'detection_results.json')
    
    if os.path.exists(results_file):
        return send_file(results_file, as_attachment=True, download_name=f'enhanced_detection_results_{session_id}.json')
    else:
        return "Results file not found", 404

if __name__ == '__main__':
    print("🚀 Starting Enhanced CCTV Footage Analyzer with GPT-4o")
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("🖼️  Frames folder:", FRAMES_FOLDER)
    print("🎯 Annotated folder:", ANNOTATED_FOLDER)
    print("📊 Results folder:", RESULTS_FOLDER)
    print("🤖 GPT-4o integration:", "Enabled" if OPENAI_API_KEY != "your-actual-openai-api-key-here" else "Disabled (Configure API key)")
    print("\n🌐 Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)