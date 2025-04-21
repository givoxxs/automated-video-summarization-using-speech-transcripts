import logging
import os
import traceback
import moviepy as mp
import whisper
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import subprocess
import cv2
import shlex

# from asyncio import run_in_executor
import asyncio
from moviepy import VideoFileClip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_audio_file(video_path: str, output_path: str) -> bool:
    video = None
    audio = None
    try:
        logger.info(f"Attempting to extract audio from: {video_path}")
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        if audio:
            logger.info(f"Writing audio to: {output_path} with codec pcm_s16le")
            audio.write_audiofile(output_path, codec='libmp3lame', logger=None) # output like: audio.mp3
            logger.info(f"Successfully created audio file: {output_path}")
            return True
        else:
            logger.warning(f"Video file {video_path} does not contain an audio track.")
            return False
    except Exception as e:
        logger.error(f"Error creating audio file from {video_path} to {output_path}: {e}", exc_info=True) 
        return False
    finally:
        if audio:
            try:
                audio.close()
            except Exception as e_close:
                 logger.error(f"Error closing audio object: {e_close}")
        if video:
            try:
                video.close()
            except Exception as e_close:
                 logger.error(f"Error closing video object: {e_close}")
                 
def load_whisper_model(model_name: str = "base") -> Optional[Any]:
    model = None
    try:
        logger.info(f"Loading Whisper model ('{model_name}')... This might take a while.")
        # model = whisper.load_model(model_name, device="cuda")
        model = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{model_name}': {e}", exc_info=True)
    return model

def speech_recognition(audio_path: str, whisper_model: Any) -> List[Tuple[str, float, float]]:
    if whisper_model is None:
        logger.error("Whisper model is not loaded. Cannot perform speech recognition.")
        return []
    
    logger.info(f"Starting speech recognition with word timestamps for: {audio_path}")
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found at path: {audio_path}")
        return [] # Trả về list rỗng khi có lỗi
    try:
        logger.info(f"Transcribing audio file with word timestamps: {audio_path}...")
        # Thêm word_timestamps=True để lấy dấu thời gian cho từng từ
        result = whisper_model.transcribe(audio_path, word_timestamps=True, verbose=False) # verbose=False để bớt log từ whisper
        logger.info(f"Transcription complete for {audio_path}.")

        # 3. Trích xuất thông tin word, start, end từ kết quả
        word_timestamps = []
        if 'segments' in result:
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        if isinstance(word_info, dict):
                            word = word_info.get('word', '').strip() # Lấy từ và loại bỏ khoảng trắng thừa
                            start = word_info.get('start')
                            end = word_info.get('end')
                        else:
                            try:
                                word = str(word_info[0]).strip()
                                start = float(word_info[1])
                                end = float(word_info[2])
                            except (IndexError, TypeError, ValueError):
                                logger.warning(f"Unexpected format for word_info: {word_info}")
                                continue # Bỏ qua word_info không hợp lệ

                        if word and start is not None and end is not None:
                            word_timestamps.append((word, round(start, 3), round(end, 3))) # Làm tròn số giây
                        else:
                            logger.warning(f"Skipping incomplete word data: {word_info}")
                else:
                    logger.warning(f"Segment starting at {segment.get('start')} does not contain word timestamps.")
        else:
            logger.warning("Transcription result does not contain 'segments'. Cannot extract timestamps.")

        logger.info(f"Extracted {len(word_timestamps)} word timestamps.")
        return word_timestamps

    except Exception as e:
        logger.error(f"Error during Whisper speech recognition for {audio_path}: {e}", exc_info=True)
        return [] # Trả về list rỗng khi có lỗ
                   
async def cut_segment_refactored(input_path: Path, start_seconds: float, end_seconds: float, output_path: Path) -> bool:
    loop = asyncio.get_running_loop()

    def _do_cut():
        logger.info(f"Đang thử cắt: {input_path} [{start_seconds:.3f}s -> {end_seconds:.3f}s] -> {output_path}")
        clip_to_write = None # Khởi tạo là None
        temp_audio_path = None # Đường dẫn file audio tạm
        try:
            # *** Sử dụng câu lệnh 'with' để quản lý tài nguyên ***
            with VideoFileClip(str(input_path)) as original_clip:
                # Kiểm tra sự tồn tại của audio một cách tin cậy
                has_audio = original_clip.audio is not None
                logger.info(f"Đã tải clip gốc. Thời lượng: {original_clip.duration:.3f}s, Phát hiện audio: {has_audio}")

                # Đảm bảo thời gian bắt đầu/kết thúc nằm trong giới hạn (tùy chọn nhưng nên làm)
                actual_end = min(end_seconds, original_clip.duration)
                actual_start = min(start_seconds, actual_end)

                # Kiểm tra thời lượng subclip hợp lệ (phải > 0)
                if actual_end - actual_start <= 0.01: # Ngưỡng nhỏ để tránh lỗi
                     logger.warning(f"Bỏ qua segment do thời gian không hợp lệ sau khi điều chỉnh: start={actual_start:.3f}, end={actual_end:.3f}")
                     return False # Không thể tạo clip có thời lượng bằng 0 hoặc âm

                logger.info(f"Đang tạo subclip từ {actual_start:.3f}s đến {actual_end:.3f}s")
                # Quan trọng: Tạo subclip *trước* khi đóng original_clip bằng 'with'
                clip_to_write = original_clip.subclipped(actual_start, actual_end)

                # Kiểm tra audio của subclip - đôi khi subclip có thể mất đối tượng audio nếu thời lượng quá nhỏ
                subclip_has_audio = clip_to_write.audio is not None
                logger.info(f"Subclip đã tạo. Thời lượng: {clip_to_write.duration:.3f}s, Có audio: {subclip_has_audio}")

                # Chỉ định rõ codec để tương thích tốt hơn
                # Sử dụng threads=1 ban đầu để ổn định, sau đó tăng nếu cần
                # Tạo tên file audio tạm duy nhất
                temp_audio_filename = f"{output_path.stem}_temp_audio_{os.urandom(4).hex()}.m4a"
                temp_audio_path = output_path.parent / temp_audio_filename

                common_args = {
                    "codec": "libx264",          # Codec video phổ biến
                    "audio_codec": "aac",       # Codec audio phổ biến
                    "temp_audiofile": str(temp_audio_path), # File audio tạm duy nhất
                    "remove_temp": True,        # Tự động xóa file audio tạm , False để giữ lại
                    "logger": None,             # Đặt là 'bar' để xem tiến trình, None để log gọn hơn
                    "threads": 4,               # Số luồng cho ffmpeg (điều chỉnh nếu cần)
                    "preset": "medium",         # Cân bằng tốc độ mã hóa/nén
                    "ffmpeg_params": ["-map_metadata", "-1", "-vsync", "cfr"] # Tránh lỗi metadata, đảm bảo fps ổn định
                }

                if subclip_has_audio:
                    logger.info(f"Đang ghi segment có audio vào {output_path}...")
                    clip_to_write.write_videofile(str(output_path), **common_args)
                else:
                    logger.warning(f"Đang ghi segment KHÔNG CÓ audio vào {output_path}...")
                    # Ghi không cần các tham số audio
                    clip_to_write.write_videofile(
                        str(output_path),
                        codec=common_args["codec"],
                        audio=False, # Tắt audio
                        logger=common_args["logger"],
                        threads=common_args["threads"],
                        preset=common_args["preset"],
                        ffmpeg_params=common_args["ffmpeg_params"]
                     )

            logger.info(f"Đã cắt segment thành công vào {output_path}")
            return True # Thành công

        except Exception as e:
            # Ghi lại toàn bộ traceback để debug chi tiết
            logger.error(f"--- Lỗi khi cắt segment {start_seconds:.3f}s-{end_seconds:.3f}s vào {output_path} ---")
            logger.error(f"Loại lỗi: {type(e).__name__}")
            logger.error(f"Chi tiết lỗi: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}") # Ghi lại toàn bộ traceback

            # Dọn dẹp file có thể bị lỗi/chưa hoàn chỉnh
            if output_path.exists():
                try:
                    output_path.unlink()
                    logger.info(f"Đã xóa file output có thể chưa hoàn chỉnh: {output_path}")
                except Exception as del_e:
                    logger.error(f"Không thể xóa file chưa hoàn chỉnh {output_path}: {del_e}")
            return False # Thất bại
        finally:
             # Đảm bảo subclip được đóng nếu nó đã được tạo
             if clip_to_write is not None and hasattr(clip_to_write, 'close'):
                 clip_to_write.close()
                 logger.debug(f"Đã đóng đối tượng subclip cho {output_path}")

             # Xóa file audio tạm nếu còn tồn tại (phòng trường hợp remove_temp=False hoặc lỗi)
             if temp_audio_path and temp_audio_path.exists():
                 try:
                     temp_audio_path.unlink()
                     logger.debug(f"Đã xóa file audio tạm: {temp_audio_path}")
                 except Exception as del_audio_e:
                     logger.warning(f"Không thể xóa file audio tạm {temp_audio_path}: {del_audio_e}")

             logger.debug(f"Hoàn tất xử lý cắt cho {output_path}")
             # 'original_clip' được đóng tự động bởi câu lệnh 'with'

    # Chạy hoạt động moviepy chặn (blocking) trong một thread pool executor
    success = await loop.run_in_executor(None, _do_cut)
    return success
             
async def concatenate_segments_ffmpeg(segment_paths: List[Path], output_path: Path) -> bool:
    """
    Ghép nối nhiều phân đoạn video thành một video tổng hợp sử dụng FFmpeg concat demuxer.

    Args:
        segment_paths: Danh sách các đường dẫn đến file video phân đoạn.
        output_path: Đường dẫn file đầu ra tổng hợp.

    Returns:
        bool: True nếu ghép nối thành công, False nếu thất bại.
    """
    # 1. Kiểm tra đầu vào và lọc file hợp lệ (tương tự phiên bản MoviePy)
    if not segment_paths:
        logger.error("No segment paths provided for concatenation.")
        return False

    valid_segment_paths = []
    for path in segment_paths:
        if not path.exists(): # Dùng path.exists() thay os.path.exists
            logger.warning(f"Segment file not found: {path}, skipping...")
            continue
        try:
             # Kiểm tra kích thước file > 0 để tránh lỗi với file rỗng
             if path.stat().st_size == 0:
                 logger.warning(f"Segment file is empty: {path}, skipping...")
                 continue
        except OSError as e:
             logger.warning(f"Could not get size for segment file {path}: {e}, skipping...")
             continue

        valid_segment_paths.append(path)

    if not valid_segment_paths:
        logger.error("No valid (existing, non-empty) segment files found for concatenation.")
        return False

    logger.info(f"Concatenating {len(valid_segment_paths)} valid segments into {output_path} using FFmpeg concat demuxer.")

    # 2. Đảm bảo thư mục chứa output_path tồn tại
    try:
        output_path.parent.mkdir(exist_ok=True, parents=True)
    except OSError as e:
         logger.error(f"Failed to create output directory {output_path.parent}: {e}")
         return False

    # 3. Tạo file danh sách tạm thời cho FFmpeg
    # Đặt tên file tạm cụ thể hơn để tránh trùng lặp
    list_file_path = output_path.with_suffix('.ffmpeg_list.txt')
    loop = asyncio.get_running_loop()

    try:
        with open(list_file_path, 'w', encoding='utf-8') as f:
            for path in valid_segment_paths:
                safe_path_str = str(path.resolve()).replace("'", "'\\''")
                f.write(f"file '{safe_path_str}'\n")
        logger.debug(f"Created FFmpeg list file: {list_file_path}")

        # 4. Xây dựng câu lệnh FFmpeg
        command = [
            'ffmpeg',
            '-f', 'concat',        # Sử dụng concat demuxer
            '-safe', '0',          # Cho phép đường dẫn trong file list (cần thiết cho đường dẫn tuyệt đối/tương đối)
            '-i', str(list_file_path), # File danh sách đầu vào
            '-map', '0:v?',        # Map luồng video nếu có (?)
            '-map', '0:a?',        # Map luồng audio nếu có (?)
            '-c:v', 'libx264',     # Mã hóa lại video bằng libx264
            '-c:a', 'aac',         # Mã hóa lại audio bằng aac
            '-preset', 'fast',     # Cân bằng tốc độ/chất lượng (có thể dùng 'medium')
            '-crf', '23',          # Chất lượng video (thấp hơn = tốt hơn, lớn hơn = file nhỏ hơn)
            '-vsync', 'cfr',       # Đảm bảo FPS ổn định cho file output (thường tốt cho tương thích)
            # '-movflags', '+faststart', # Tùy chọn: tối ưu cho xem trực tuyến (ghi moov atom ở đầu)
            str(output_path)       # File đầu ra
        ]

        # Ghi lại câu lệnh sẽ chạy để dễ debug
        # Dùng shlex.join cho Python 3.8+ để hiển thị an toàn hơn, hoặc join thủ công
        try:
             import shlex
             log_command = shlex.join(command)
        except ImportError:
             log_command = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in command) # Cách join đơn giản
        logger.info(f"Running FFmpeg command: {log_command}")
        
        # 5. *** Chạy FFmpeg đồng bộ trong executor ***
        def run_ffmpeg_sync():
            # capture_output=True để lấy stdout/stderr
            # text=True để output là string (dễ decode hơn)
            # check=False để tự xử lý lỗi dựa trên returncode
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='ignore', check=False)
            return result
        # Chạy hàm đồng bộ trong executor của event loop hiện tại
        result = await loop.run_in_executor(None, run_ffmpeg_sync)

        # 6. Kiểm tra kết quả từ subprocess.run
        if result.returncode != 0:
            logger.error(f"FFmpeg concatenation failed with exit code {result.returncode}")
            logger.error(f"FFmpeg stderr:\n{result.stderr}")
            output_path.unlink(missing_ok=True)
            return False
        else:
            if not output_path.exists() or output_path.stat().st_size == 0:
                 logger.error("FFmpeg reported success (exit code 0), but the output file is missing or empty.")
                 output_path.unlink(missing_ok=True)
                 return False

            logger.info(f"FFmpeg concatenation successful: {output_path}")
            return True

    except Exception as e:
        logger.error(f"An unexpected error occurred during FFmpeg concatenation process: {e}", exc_info=True)
        # Xóa file output có thể bị lỗi/chưa hoàn chỉnh
        output_path.unlink(missing_ok=True)
        return False
    finally:
        # 7. Dọn dẹp file danh sách tạm thời (luôn thực hiện)
        if list_file_path.exists():
             try:
                 list_file_path.unlink()
                 logger.debug(f"Deleted temporary list file: {list_file_path}")
             except Exception as e_unlink:
                 # Ghi lại cảnh báo nếu không xóa được file tạm, nhưng không làm hàm thất bại
                 logger.warning(f"Could not delete temporary list file {list_file_path}: {e_unlink}")
                                