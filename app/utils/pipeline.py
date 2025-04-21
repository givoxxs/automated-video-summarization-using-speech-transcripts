# bao gồm video cần summary và thời lượng video sau khi meeting xong, ví dụ 3p, 5p, 7p.
from pathlib import Path
import requests
from typing import Dict, Any, Optional, List, Tuple
from app.config import get_config
from app.utils.extract import (
    create_audio_file, 
    extract_transcript, 
    load_whisper_model,
)
from app.models.base import TimedWord, Segment

from app.utils.segmentation import segment_transcript
from app.utils.calc_score import calc_score_segments
from app.utils.skim_generator import generate_skim 
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

config = get_config()

model_whisper = load_whisper_model(model_name=config.WHISPER_MODEL_NAME)
def get_model_whisper() -> Optional[Any]:
    return model_whisper

WHISPER_API_URL = os.getenv("WHISPER_API_URL", "")
WHISPER_API_URL = WHISPER_API_URL.rstrip("/")  # Đảm bảo không có dấu "/" ở cuối URL
WHISPER_API_URL = WHISPER_API_URL + "/transcribe"  # Thêm /transcribe vào cuối URL nếu cần

def call_whisper_api(audio_path: Path) -> List[TimedWord]:
    """Hàm gọi API Whisper và trả về List[TimedWord]."""
    logger.info(f"Calling Whisper API at: {WHISPER_API_URL} for audio: {audio_path}")
    transcripts_data = []
    try:
        # Mở file audio ở chế độ đọc nhị phân (rb)
        with open(audio_path, 'rb') as f:
            # Chuẩn bị file để gửi
            files = {'file': (audio_path.name, f, 'audio/wav')} # Cung cấp tên file và content type nếu muốn

            # Gửi yêu cầu POST đến API
            response = requests.post(WHISPER_API_URL, files=files, timeout=300) # Tăng timeout nếu cần

            # Kiểm tra lỗi HTTP (ví dụ: 4xx, 5xx)
            response.raise_for_status()

            # Lấy kết quả JSON từ API (đây là list các dictionary)
            results_list_dict = response.json()

            # Chuyển đổi list[dict] thành list[TimedWord]
            # (Bước này cần thiết nếu các hàm sau như segment_transcript yêu cầu đối tượng TimedWord)
            transcripts_data = [TimedWord(**item) for item in results_list_dict]
            logger.info(f"API call successful. Received {len(transcripts_data)} words.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Whisper API: {e}")
    except Exception as e:
        logger.error(f"An error occurred processing the API response: {e}")

    return transcripts_data

async def summary_video(
    video_path: Path,
    target_duration: int = 600, # 10 phút
    model_name: str = "tiny",
):
    # step 1: extract video
    try: 
        video_name = video_path.stem # stem là tên file không có đuôi
        logger.info(f"Step 1: Extracting audio from video {video_name}...")
        
        if config.USE_AZURE_SPEECH:
            logger.info("Using Azure Speech Service for audio extraction.")
            output_path = config.audio_path / f"{video_name}_audio.wav"
        else:
            logger.info("Using local audio extraction.")
            output_path = config.audio_path / f"{video_name}_audio.mp3"
            
        logger.info(f"Output path: {output_path}")
        # Kiểm tra xem file đã tồn tại chưa
        if output_path.exists():
            logger.info(f"Audio file already exists at {output_path}.")
        response_au = create_audio_file(video_path, output_path)
        # response_au = True
        logger.info(f"Audio file created at {output_path}.")
        
        if not response_au:
            logger.error("Failed to create audio file.")
            return
        
        if config.WHISPER_LOCAL or WHISPER_API_URL == "":
            model = get_model_whisper()
            if not model:
                logger.error("Failed to load Whisper model.")
                return
            transcripts = extract_transcript(output_path, model)
            if not transcripts:
                logger.error("Failed to extract transcript.")
                return
        else:
            # Nếu không sử dụng Whisper local, gọi API để lấy transcript
            logger.info("Using Whisper API for transcript extraction.")
            transcripts = call_whisper_api(output_path)
            if not transcripts:
                logger.error("Failed to extract transcript via API.")
                return
            
        # step 4: segment transcript
        logger.info("Step 4: Segmenting transcript...")
        segments = await segment_transcript(transcripts)
        if not segments:
            logger.error("Failed to segment transcript.")
            return
        logger.info(f"Number of segments: {len(segments)}")
        
        # step 5: calculate score for segments
        logger.info("Step 5: Calculating scores for segments...")
        scored_segments = await calc_score_segments(segments)
        if not scored_segments:
            logger.error("Failed to calculate scores for segments.")
            return
        
        # step 6: generate skim
        logger.info("Step 6: Generating skim...")
        final_summary_path = await generate_skim(
            segments=scored_segments,
            target_duration=target_duration,
            original_video_path=video_path,
            output_filename_base=video_name,
        )
        if not final_summary_path:
            logger.error("Failed to generate skim.")
            return
        logger.info("Skim generated successfully.")
        logger.info(f"Final summary path: {final_summary_path}")
        
        # step 7: clear temp files
        logger.info("Step 7: Cleaning up temporary files...")
        temp_files = [config.audio_path / f"{video_name}_audio.wav", config.audio_path / f"{video_name}_audio.mp3"]
        for temp_file in temp_files:
            if temp_file.exists():
                os.remove(temp_file)
                logger.info(f"Removed temporary file: {temp_file}")
            else:
                logger.info(f"Temporary file not found: {temp_file}")
                
        # remove temp_list.txt, ở ngoài cùng chương trình 
        try:
            os.remove("./temp_list.txt")
        except FileNotFoundError:
            logger.info("Temporary list file not found: temp_list.txt")
        logger.info("Removed temporary list file: temp_list.txt")
        
        # step 8: remove video file
        if video_path.exists():
            os.remove(video_path)
            logger.info(f"Removed original video file: {video_path}")
        else:
            logger.info(f"Original video file not found: {video_path}")
            
        logger.info("Cleanup completed.")
        # Trả về đường dẫn của video đã tóm tắt        
        return final_summary_path
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
    