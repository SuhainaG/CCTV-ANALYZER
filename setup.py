#!/usr/bin/env python3
"""
Setup script for AI Video Analysis Platform
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg detected")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg not found. Audio processing will be disabled.")
        print("To enable audio processing, install FFmpeg:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - macOS: brew install ffmpeg")
        print("  - Ubuntu: sudo apt install ffmpeg")
        return False

def install_requirements():
    """Install Python requirements"""
    try:
        print("Installing Python dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'extracted_frames',
        'annotated_frames',
        'results',
        'audio_processing'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_environment():
    """Setup environment variables"""
    if not os.path.exists('.env'):
        print("Creating .env file...")
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# AssemblyAI API Configuration
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# File Upload Configuration
MAX_CONTENT_LENGTH=1073741824  # 1GB in bytes

# Processing Configuration
SIMILARITY_THRESHOLD=0.85
MIN_FRAME_INTERVAL=30
CONFIDENCE_THRESHOLD=0.25

# TTS Configuration
TTS_RATE=180
TTS_VOLUME=0.8
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ Created .env file - Please update with your API keys")
    else:
        print("✓ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up AI Video Analysis Platform")
    print("=" * 50)
    
    check_python_version()
    ffmpeg_available = check_ffmpeg()
    install_requirements()
    create_directories()
    setup_environment()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Update .env file with your API keys:")
    print("   - OpenAI API key for GPT-4o analysis")
    print("   - AssemblyAI API key for audio processing")
    print("2. Run the application: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    
    if not ffmpeg_available:
        print("\nNote: Install FFmpeg to enable audio processing features")

if __name__ == "__main__":
    main()
