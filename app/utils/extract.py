import os
import moviepy as mp
import whisper
from typing import List, Tuple, Dict, Any, Optional
from app.models.base import TimedWord
import asyncio
from moviepy import VideoFileClip, concatenate_videoclips
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_audio_file(video_path: str, output_path: str) -> bool:
    video = None
    audio = None
    try:
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        if audio:
            audio.write_audiofile(output_path, codec='libmp3lame', logger=None) # output like: audio.mp3
            return True
        else:
            print(f"Video file {video_path} does not contain an audio track.")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        if audio:
            try:
                audio.close()
            except Exception as e_close:
                print(f"Error closing audio object: {e_close}")
                pass
        if video:
            try:
                video.close()
            except Exception as e_close:
                print(f"Error closing video object: {e_close}")
                pass
            
def load_whisper_model(model_name: str = "base") -> Optional[Any]:
    model = None
    try:
        # model = whisper.load_model(model_name, device="cuda")
        model = whisper.load_model(model_name)
    except Exception as e:
        print(f"Failed to load Whisper model '{model_name}': {e}")
    return model

def extract_transcript(audio_path: str, whisper_model: Any) -> List[TimedWord]:
    if whisper_model is None:
        print("Whisper model is not loaded. Cannot perform speech recognition.")
        return []
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found at path: {audio_path}")
        return [] 
    try:
        result = whisper_model.transcribe(audio_path, word_timestamps=True)
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
        return transcripts
    except Exception as e:
        print(f"An error occurred during speech recognition: {e}")
        return [] 
    
async def cut_segment(video_path: Path, start_time: float, end_time: float, output_path: Path) -> bool:
    """
    Cắt một phân đoạn từ video gốc dựa trên thời gian bắt đầu và kết thúc.
    
    Args:
        video_path: Đường dẫn đến file video gốc
        start_time: Thời gian bắt đầu (giây)
        end_time: Thời gian kết thúc (giây)
        output_path: Đường dẫn file đầu ra
        
    Returns:
        bool: True nếu cắt thành công, False nếu thất bại
    """
    video = None
    try:
        logger.info(f"Cutting segment from {start_time:.2f}s to {end_time:.2f}s from {video_path}")
        
        # Sử dụng thread executor để thực hiện thao tác cắt video không block event loop
        loop = asyncio.get_event_loop()
        video = await loop.run_in_executor(
            None, 
            lambda: VideoFileClip(str(video_path), target_resolution=None).subclip(start_time, end_time)
        )
        
        # Ghi file video đã cắt
        await loop.run_in_executor(
            None,
            lambda: video.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=f"{str(output_path)}_temp_audio.m4a",
                remove_temp=True,
                logger=None
            )
        )
        
        logger.info(f"Successfully cut segment to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error cutting segment from {start_time:.2f}s to {end_time:.2f}s: {e}", exc_info=True)
        return False
    
    finally:
        if video:
            try:
                video.close()
            except Exception as e:
                logger.error(f"Error closing video clip: {e}")

async def concatenate_segments(segment_paths: List[Path], output_path: Path) -> bool:
    """
    Ghép nối nhiều phân đoạn video thành một video tổng hợp.
    
    Args:
        segment_paths: Danh sách các đường dẫn đến file video phân đoạn
        output_path: Đường dẫn file đầu ra tổng hợp
        
    Returns:
        bool: True nếu ghép nối thành công, False nếu thất bại
    """
    clips = []
    try:
        if not segment_paths:
            logger.error("No segment paths provided for concatenation.")
            return False
            
        logger.info(f"Concatenating {len(segment_paths)} segments into {output_path}")
        
        # Tải tất cả các clip
        loop = asyncio.get_event_loop()
        for path in segment_paths:
            if not os.path.exists(path):
                logger.warning(f"Segment file not found: {path}, skipping...")
                continue
                
            clip = await loop.run_in_executor(None, lambda: VideoFileClip(str(path)))
            clips.append(clip)
            
        if not clips:
            logger.error("No valid video clips to concatenate.")
            return False
            
        # Ghép nối các clip
        final_clip = concatenate_videoclips(clips)
        
        # Ghi file video tổng hợp
        await loop.run_in_executor(
            None,
            lambda: final_clip.write_videofile(
                str(output_path),
                codec="libx264", 
                audio_codec="aac",
                temp_audiofile=f"{str(output_path)}_temp_audio.m4a",
                remove_temp=True,
                logger=None
            )
        )
        
        logger.info(f"Successfully concatenated segments to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error concatenating segments: {e}", exc_info=True)
        return False
        
    finally:
        # Đóng tất cả các clips để giải phóng tài nguyên
        for clip in clips:
            try:
                clip.close()
            except Exception as e:
                logger.error(f"Error closing video clip: {e}")