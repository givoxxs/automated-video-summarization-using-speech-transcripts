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

config = get_config()

model_whisper = load_whisper_model(model_name=config.WHISPER_MODEL_NAME)
def get_model_whisper() -> Optional[Any]:
    return model_whisper
WHISPER_API_URL = "https://899d-34-124-163-238.ngrok-free.app/transcribe"

def call_whisper_api(audio_path: Path) -> List[TimedWord]:
    """Hàm gọi API Whisper và trả về List[TimedWord]."""
    print(f"Calling Whisper API at: {WHISPER_API_URL} for audio: {audio_path}")
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
            print(f"API call successful. Received {len(transcripts_data)} words.")

    except requests.exceptions.RequestException as e:
        print(f"Error calling Whisper API: {e}")
    except Exception as e:
        print(f"An error occurred processing the API response: {e}")

    return transcripts_data

async def summary_video(
    video_path: Path,
    target_duration: int = 300, # 5 phút
    model_name: str = "tiny",
):
    # step 1: extract video
    try: 
        video_name = video_path.stem # stem là tên file không có đuôi
        print(f"Step 1: Extracting audio from video {video_name}...")
        
        output_path = config.audio_path / f"{video_name}_audio.mp3"
        response_au = create_audio_file(video_path, output_path)
        
        if not response_au:
            print("Failed to create audio file.")
            return
        # step 2: load whisper model
        model = get_model_whisper()
        if not model:
            print("Failed to load Whisper model.")
            return
        # step 3: extract transcript
        transcripts = extract_transcript(output_path, model)
        if not transcripts:
            print("Failed to extract transcript.")
            return
        
        # --- Step 3: Gọi API để extract transcript ---
        # print("Step 3: Extracting transcript via API...")
        # transcripts = call_whisper_api(output_path) # Gọi hàm mới
        # if not transcripts:
        #     print("Failed to extract transcript via API.")
        #     # Có thể thêm xử lý xóa file audio tạm ở đây nếu muốn
        #     # if os.path.exists(audio_path): os.remove(audio_path)
        #     return
        # step 4: segment transcript
        print("Step 4: Segmenting transcript...")
        segments = await segment_transcript(transcripts)
        if not segments:
            print("Failed to segment transcript.")
            
            return
        print(f"Number of segments: {len(segments)}")
        
        # step 5: calculate score for segments
        print("Step 5: Calculating scores for segments...")
        scored_segments =  await calc_score_segments(segments)
        if not scored_segments:
            print("Failed to calculate scores for segments.")
            return
        # step 6: generate skim
        print("Step 6: Generating skim...")
        final_summary_path = await generate_skim(
            segments=scored_segments,
            target_duration=target_duration,
            original_video_path=video_path,
            output_filename_base=video_name,
        )
        if not final_summary_path:
            print("Failed to generate skim.")
            return
        print("Skim generated successfully.")
        print(f"Final summary path: {final_summary_path}")
        # step 7: save skim
        print("Step 7: Saving skim...")
        with open(final_summary_path, 'w') as f:
            f.write(final_summary_path.read_text())
        print(f"Skim saved to {final_summary_path}.")
    except Exception as e:
        print(f"Error processing video: {e}")
    