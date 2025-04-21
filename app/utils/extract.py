# filepath: d:\Sgroup\Sgroup-AI\video-meet-summarier\app\utils\extract.py
import os
import moviepy as mp
import whisper
from typing import List, Tuple, Dict, Any, Optional
from app.models.base import TimedWord
import asyncio
from moviepy import VideoFileClip, concatenate_videoclips
import logging
from pathlib import Path
import time
import wave
import io
import json

# Azure Speech Service imports
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_SDK_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_SDK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Azure Speech SDK is not available. Install it with: pip install azure-cognitiveservices-speech")

from app.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = get_config()

def create_audio_file(video_path: str, output_path: str) -> bool:
    video = None
    audio = None
    try:
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        if audio:
            # Chuyển đổi output_path thành chuỗi nếu nó là đối tượng Path
            output_path_str = str(output_path)
            
            # Kiểm tra nếu output_path kết thúc bằng .mp3, đổi thành .wav cho Azure
            if output_path_str.lower().endswith('.mp3') and config.USE_AZURE_SPEECH:
                output_path_str = output_path_str[:-4] + '.wav'
                logger.info(f"Changed output format to WAV for Azure compatibility: {output_path_str}")
                audio.write_audiofile(output_path_str, codec='pcm_s16le', ffmpeg_params=["-ac", "1", "-ar", "16000"], logger=None)
            else:
                audio.write_audiofile(output_path_str, codec='libmp3lame', logger=None)  # output like: audio.mp3
            return True
        else:
            logger.warning(f"Video file {video_path} does not contain an audio track.")
            return False
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return False
    finally:
        if audio:
            try:
                audio.close()
            except Exception as e_close:
                logger.error(f"Error closing audio object: {e_close}")
                pass
        if video:
            try:
                video.close()
            except Exception as e_close:
                logger.error(f"Error closing video object: {e_close}")
                pass
            
def load_whisper_model(model_name: str = "base") -> Optional[Any]:
    model = None
    try:
        # model = whisper.load_model(model_name, device="cuda")
        model = whisper.load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{model_name}': {e}", exc_info=True)
    return model

def extract_transcript_whisper(audio_path: str | Path, whisper_model: Any) -> List[TimedWord]:
    """
    Extract transcript using local Whisper model
    """
    if whisper_model is None:
        logger.error("Whisper model is not loaded. Cannot perform speech recognition.")
        return []
    
    # Convert Path object to string if needed
    audio_path_str = str(audio_path) if isinstance(audio_path, Path) else audio_path
    
    if not os.path.exists(audio_path_str):
        logger.error(f"Audio file not found at path: {audio_path_str}")
        return [] 
    try:
        logger.info(f"Processing audio with Whisper model: {audio_path_str}")
        result = whisper_model.transcribe(audio_path_str, word_timestamps=True)
        segments = result["segments"]

        transcripts = []
        for segment in segments:
            words = segment["words"]
            for word in words:
                word_info = TimedWord(
                    word=word["word"],
                    start=word["start"],
                    end=word["end"]
                )
                transcripts.append(word_info)
        logger.info(f"Whisper transcription completed. Found {len(transcripts)} words.")
        return transcripts
    except Exception as e:
        logger.error(f"An error occurred during Whisper speech recognition: {e}", exc_info=True)
        return []

def extract_transcript_azure(audio_path: str | Path) -> List[TimedWord]:
    """
    Extract transcript using Azure Speech Service
    """
    if not AZURE_SPEECH_SDK_AVAILABLE:
        logger.error("Azure Speech SDK is not available. Cannot perform speech recognition.")
        return []
    
    # Convert Path object to string if needed
    audio_path_str = str(audio_path) if isinstance(audio_path, Path) else audio_path
    
    if not os.path.exists(audio_path_str):
        logger.error(f"Audio file not found at path: {audio_path_str}")
        return []
        
    try:
        logger.info(f"Processing audio with Azure Speech Service: {audio_path_str}")
        
        # Get Azure credentials from config
        speech_key = config.AZURE_SPEECH_KEY
        service_region = config.AZURE_SPEECH_REGION
        
        if not speech_key:
            logger.error("Azure Speech Service key is not configured. Check your environment variables.")
            return []
            
        # Configure speech config
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.request_word_level_timestamps()
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1000")
        speech_config.enable_audio_logging()
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        # Configure audio config
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path_str)
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            language=config.AZURE_SPEECH_LANGUAGE
        )
        
        # Process recognition
        words_with_timestamps = []
        
        def process_result(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.debug(f"RECOGNIZED: {evt.result.text}")
                
                if hasattr(evt.result, 'json'):
                    result_json = json.loads(evt.result.json)
                    
                    if 'NBest' in result_json and len(result_json['NBest']) > 0:
                        # Get best result
                        best_result = result_json['NBest'][0]
                        
                        if 'Words' in best_result:
                            words = best_result['Words']
                            for word in words:
                                offset_sec = word['Offset'] / 10000000  # Convert from 100-nanosecond units to seconds
                                duration_sec = word['Duration'] / 10000000
                                
                                word_info = TimedWord(
                                    word=word['Word'],
                                    start=offset_sec,
                                    end=offset_sec + duration_sec
                                )
                                words_with_timestamps.append(word_info)
            
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning(f"NOMATCH: {evt.result.no_match_details}")
        
        done = False
        
        def stop_cb(evt):
            nonlocal done
            logger.info(f"Speech recognition stopped: {evt}")
            done = True
        
        # Connect callbacks
        speech_recognizer.recognized.connect(process_result)
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)
        
        # Start recognition
        logger.info("Starting continuous speech recognition")
        speech_recognizer.start_continuous_recognition()
        
        # Wait for recognition to complete
        while not done:
            time.sleep(0.5)
        
        logger.info(f"Azure transcription completed. Found {len(words_with_timestamps)} words.")
        return sorted(words_with_timestamps, key=lambda x: x.start)
        
    except Exception as e:
        logger.error(f"An error occurred during Azure speech recognition: {e}", exc_info=True)
        return []

def extract_transcript(audio_path: str | Path, whisper_model: Any = None) -> List[TimedWord]:
    """
    Extracts transcript from audio, using either Whisper or Azure Speech Service
    """
    if config.USE_AZURE_SPEECH:
        logger.info("Using Azure Speech Service for speech recognition")
        return extract_transcript_azure(audio_path)
    else:
        logger.info("Using Whisper for speech recognition")
        return extract_transcript_whisper(audio_path, whisper_model)