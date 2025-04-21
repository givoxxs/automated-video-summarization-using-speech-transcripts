# Video Meeting Summarizer

![Video Meeting Summarizer](https://img.shields.io/badge/SGroup--AI-Video%20Meeting%20Summarizer-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0%2B-blue)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-blue)

## 📝 Overview

Video Meeting Summarizer is an AI-powered web application that automatically condenses long meeting recordings into concise summary videos. The tool uses speech recognition, natural language processing, and video processing techniques to identify and extract the most important segments of a meeting, saving time for busy professionals.

## ✨ Features

- **Video Upload**: Easy drag & drop or file selection interface
- **Customizable Summary Duration**: Choose your preferred summary length
- **Advanced Processing**: Multi-stage pipeline for accurate summarization
  - Audio extraction from video
  - Speech-to-text transcription using Whisper
  - Transcript segmentation
  - Importance scoring of segments
  - Intelligent video skimming
- **Real-time Progress Tracking**: Monitor the processing stages with a progress bar
- **Instant Playback**: Watch your summarized video directly in the browser
- **Download Option**: Save the summarized video to your device

## 🛠️ Technical Architecture

The application is built with a modern tech stack:

- **Frontend**: HTML, CSS, JavaScript with Bootstrap 5
- **Backend**: Python with FastAPI
- **Speech Recognition**: OpenAI's Whisper model
- **Video Processing**: MoviePy for video manipulation
- **Asynchronous Processing**: Background tasks for handling resource-intensive operations

### Processing Pipeline

1. **Video Upload**: User submits a video file through the web interface
2. **Audio Extraction**: The system extracts audio from the video
3. **Transcription**: Whisper API converts speech to text with timestamps
4. **Segmentation**: The transcript is divided into meaningful segments
5. **Scoring**: Each segment is scored based on importance
6. **Video Skimming**: Important segments are selected to fit the target duration
7. **Compilation**: Selected segments are combined into a cohesive summary video

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- FFmpeg (for video processing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-meet-summarizer.git
   cd video-meet-summarizer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```bash
   # Update config.py with your settings if needed
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## 📁 Project Structure

```
video-meet-summarier/
├── app/
│   ├── apis/             # API endpoints
│   ├── models/           # Data models
│   ├── static/           # CSS, JavaScript, etc.
│   ├── templates/        # HTML templates
│   ├── utils/            # Utility functions
│   ├── config.py         # Application configuration
│   └── main.py           # Application entry point
├── data/
│   ├── audio/            # Extracted audio files
│   ├── summaries/        # Generated summary files
│   └── video/            # Uploaded video files
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🔧 Configuration

The application can be configured by modifying `app/config.py`. Key settings include:

- `WHISPER_MODEL_NAME`: Model size for the Whisper speech recognition (tiny, base, small, medium, large)
- `WHISPER_API_URL`: URL for external Whisper API integration
- File paths for audio, video, and summary storage

## 📊 How It Works

### 1. Speech Recognition
The application uses OpenAI's Whisper model to transcribe speech from the video with precise timestamps. Each word is assigned a start and end time.

### 2. Segmentation
The transcript is divided into meaningful segments based on linguistic and semantic patterns. This creates logical units for analysis.

### 3. Scoring Algorithm
Each segment is scored based on:
- Information density
- Presence of key terms or phrases
- Speaker identification (if available)
- Linguistic features indicating importance

### 4. Video Skimming
The system selects segments with the highest scores while ensuring the summary meets the target duration and maintains narrative coherence.

### 5. Video Processing
Selected segments are extracted from the original video and concatenated to create a seamless summary video.

## ⚠️ Troubleshooting

### Common Issues

- **FFmpeg Missing**: Ensure FFmpeg is installed and available in your system PATH
- **Memory Errors**: Large videos may require significant RAM; consider processing in smaller chunks
- **Transcription Accuracy**: Speech recognition accuracy varies with audio quality; ensure clear audio for best results

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com)
