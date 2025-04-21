# Video Meeting Summarizer

![Video Meeting Summarizer](https://img.shields.io/badge/Status-In%20Development-yellow)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0+-green)
![Whisper](https://img.shields.io/badge/Whisper-AI-purple)

## 📑 Overview

Video Meeting Summarizer is an advanced tool that automatically condenses long videos (meetings, lectures, presentations) into concise summaries by analyzing speech transcripts. Inspired by the research paper ["Automated video summarization using speech transcripts"](https://scispace.com/pdf/automated-video-summarization-using-speech-transcripts-51swqrqbg4.pdf), this project implements an intelligent algorithm that segments videos based on speech patterns and ranks segments by content importance.

### 🎯 Target Video Types

- **Documentaries**: Condense long documentary films while preserving key information
- **Educational Videos**: Summarize lectures and instructional videos for easier review
- **Presentations**: Create concise versions of conference talks and business presentations
- **Meeting Recordings**: Distill important content from lengthy video conference calls

## 🧠 How It Works

The summarization algorithm follows a comprehensive pipeline:

1. **Audio Extraction**: Extract audio track from the video file
2. **Speech Recognition**: Generate accurate transcripts using Whisper AI
3. **Segmentation**: Divide transcript into logical segments based on speech pauses
4. **Segment Scoring**: Rank segments by:
   - Word frequency analysis
   - Important word pair co-occurrence
   - Dominant word pair detection
5. **Video Skim Generation**: Select highest-scoring segments to create a condensed video summary
6. **Final Processing**: Concatenate selected segments into a seamless summary video

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- FFmpeg (for audio/video processing)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/video-meet-summarizer.git
cd video-meet-summarizer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Make sure FFmpeg is installed and available in your system PATH.

### Usage

#### API Endpoint

The application exposes a FastAPI endpoint for video summarization:

```
POST /api/v1/summarize
```

Parameters:
- `file`: The video file to summarize (multipart/form-data)
- `summary_type`: Type of summary to generate
- `user_id`: Optional user identifier

#### Direct Script Usage

You can also invoke the summarization pipeline directly:

```python
from app.utils.pipeline import summary_video
from pathlib import Path

# Summarize a video with default 5-minute target length
await summary_video(
    video_path=Path("path/to/your/video.mp4"),
    target_duration=300,  # 5 minutes (in seconds)
    model_name="base"     # Whisper model size
)
```

## 🛠️ Project Structure

```
video-meet-summarier/
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application entry point
│   ├── config.py        # Configuration management
│   ├── apis/            # API endpoint definitions
│   ├── models/          # Data models
│   ├── static/          # Static assets (CSS, JS)
│   ├── templates/       # HTML templates
│   └── utils/           # Utility modules
│       ├── calc_score.py       # Segment scoring
│       ├── extract.py          # Audio extraction & transcription
│       ├── pipeline.py         # Main processing pipeline
│       ├── segmentation.py     # Transcript segmentation
│       ├── skim_generator.py   # Video summary generation
│       └── video_processor.py  # Video manipulation functions
└── data/                # Data storage directory
    ├── audio/           # Extracted audio files
    ├── summaries/       # Generated video summaries
    ├── temp_skims/      # Temporary processing files
    ├── transcript/      # Generated transcripts
    └── video/           # Original uploaded videos
```

## ⚙️ Configuration

The application is configurable via the `Config` class in `app/config.py`. Key parameters include:

- **Segmentation Parameters**:
  - `SEGMENTATION_N`: Speech pause threshold (1.5 seconds default)
  - `SEGMENTATION_M`: Minimum segment length (10 seconds default)

- **Scoring Parameters**:
  - `SCORING_K`: Word frequency weight (2.0 default)
  - `SCORING_B`: Base scoring factor (0.75 default)
  - `DOMINANT_PAIR_COUNT`: Number of word pairs to consider (30 default)
  - `DOMINANT_PAIR_BOOST`: Boost factor for important pairs (1.2 default)

- **Model Configuration**:
  - `WHISPER_MODEL_NAME`: Whisper model size ("base" default)

## 🔬 Technical Details

### Segmentation Algorithm

The segmentation process divides transcripts based on natural speech pauses:
- A new segment begins when a pause exceeds threshold N (default 1.5 seconds)
- Segments shorter than threshold M (default 10 seconds) are merged
- Each segment contains word timing information for precise video cutting

### Scoring Methodology

Segments are scored using a sophisticated algorithm that considers:
1. Word frequency within the document
2. Co-occurrence patterns of important word pairs
3. Presence of dominant topic-related terms
4. Segment length normalization

### Key Components

- **Whisper API Integration**: Leverages OpenAI's Whisper for accurate speech recognition
- **Asynchronous Processing**: Uses `asyncio` for efficient audio/video processing
- **FastAPI Web Service**: Provides a modern, high-performance API interface
- **Pydantic Models**: Ensures data validation and serialization

## 🤝 Contributing

Contributions to improve Video Meeting Summarizer are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

This project is based on research from:
- ["Automated video summarization using speech transcripts"](https://scispace.com/pdf/automated-video-summarization-using-speech-transcripts-51swqrqbg4.pdf)

## 📧 Contact

For questions or feedback, please reach out to:
- Project Team - [your-email@example.com](mailto:your-email@example.com)
