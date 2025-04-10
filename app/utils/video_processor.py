import logging
import os
import moviepy as mp
import whisper
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import subprocess
import cv2


# from asyncio import run_in_executor
import asyncio
from moviepy import VideoFileClip, concatenate_videoclips

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
        return [] # Trả về list rỗng khi có lỗi
    
# async def cut_segment(video_path: Path, start_time: float, end_time: float, output_path: Path) -> bool:
#     """
#     Cắt một phân đoạn từ video gốc dựa trên thời gian bắt đầu và kết thúc.
    
#     Args:
#         video_path: Đường dẫn đến file video gốc
#         start_time: Thời gian bắt đầu (giây)
#         end_time: Thời gian kết thúc (giây)
#         output_path: Đường dẫn file đầu ra
        
#     Returns:
#         bool: True nếu cắt thành công, False nếu thất bại
#     """
#     video = None
#     loop = asyncio.get_event_loop()
#     try:
#         logger.info(f"Cutting segment from {start_time:.2f}s to {end_time:.2f}s from {video_path}")
        
#         # Sử dụng thread executor để thực hiện thao tác cắt video không block event loop
#         loop = asyncio.get_event_loop()
#         video = await loop.run_in_executor(
#             None, 
#             lambda: VideoFileClip(str(video_path), target_resolution=None).subclipped(start_time, end_time)
#         )
        
#         # Ghi file video đã cắt
#         await loop.run_in_executor(
#             None,
#             lambda: video.write_videofile(
#                 str(output_path),
#                 codec="libx264",
#                 audio_codec="aac",
#                 temp_audiofile=f"{str(output_path)}_temp_audio.m4a",
#                 remove_temp=True,
#                 logger=None
#             )
#         )
        
#         await asyncio.sleep(0.2)
        
#         logger.info(f"Successfully cut segment to {output_path}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error cutting segment from {start_time:.2f}s to {end_time:.2f}s: {e}", exc_info=True)
#         return False
    
#     finally:
#         if video:
#             try:
#                 video.close()
#             except Exception as e:
#                 logger.error(f"Error closing video clip: {e}")

async def cut_segment(video_path: Path, start_time: float, end_time: float, output_path: Path) -> bool:
    """
    Cắt một phân đoạn từ video gốc dựa trên thời gian bắt đầu và kết thúc.
    (Với điều chỉnh end_time để thử khắc phục lỗi frame cuối)
    """
    video_clip_to_write = None # Biến lưu clip sẽ được ghi
    loop = asyncio.get_event_loop()

    try:
        # Log thời gian với độ chính xác cao hơn
        logger.info(f"Cutting segment from {start_time:.3f}s to {end_time:.3f}s -> {output_path.name}")

        # --- BẮT ĐẦU: Điều chỉnh end_time ---
        adjusted_end_time = end_time
        try:
            # Lấy FPS một cách an toàn bằng cách load clip riêng
            # logger.debug(f"Getting FPS for {video_path} to adjust end time...")
            temp_clip_info = await loop.run_in_executor(None, lambda: VideoFileClip(str(video_path)))
            fps = temp_clip_info.fps
            # Đảm bảo đóng clip tạm này ngay lập tức
            try:
                temp_clip_info.close()
            except Exception:
                pass # Bỏ qua lỗi khi đóng clip tạm

            if fps and fps > 0:
                frame_duration = 1.0 / fps
                # Trừ đi nửa thời lượng frame
                potential_adjusted_end_time = end_time - (frame_duration / 2.0)
                # Đảm bảo việc điều chỉnh không làm end_time <= start_time
                if potential_adjusted_end_time > start_time:
                    adjusted_end_time = potential_adjusted_end_time
                    # logger.debug(f"Original end_time: {end_time:.3f}, Adjusted end_time: {adjusted_end_time:.3f} (FPS: {fps:.2f})")
                else:
                    # Ghi log nếu không điều chỉnh được do làm thời gian không hợp lệ
                    logger.warning(f"Time adjustment skipped for segment {start_time:.3f}-{end_time:.3f}: would make end_time <= start_time.")
            else:
                # Ghi log nếu không lấy được FPS hợp lệ
                logger.warning(f"Could not get valid FPS ({fps}) for {video_path}. Using original end_time {end_time:.3f}.")

        except Exception as e_fps:
            logger.warning(f"Exception getting FPS for time adjustment: {e_fps}. Using original end_time {end_time:.3f}.")
            # Giữ nguyên adjusted_end_time = end_time
        # --- KẾT THÚC: Điều chỉnh end_time ---

        # Load clip để cắt với end_time đã được điều chỉnh (nếu có)
        # logger.debug(f"Loading subclip {start_time:.3f}s to {adjusted_end_time:.3f}s")
        video_clip_to_write = await loop.run_in_executor(
            None,
            # Load lại clip gốc mỗi lần cắt để tránh lỗi state của clip
            lambda: VideoFileClip(str(video_path), target_resolution=None).subclipped(start_time, adjusted_end_time)
        )

        # Kiểm tra xem subclip có thành công không
        if not video_clip_to_write:
             raise ValueError(f"Subclip operation failed for {start_time:.3f}-{adjusted_end_time:.3f}")

        # Ghi file video đã cắt
        # logger.debug(f"Writing subclip to {output_path}")
        await loop.run_in_executor(
            None,
            lambda: video_clip_to_write.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=f"{str(output_path)}_temp_audio.m4a",
                remove_temp=True,
                logger=None # Giữ logger=None để tránh spam progress bar
            )
        )

        # (Tùy chọn) Thêm độ trễ nhỏ nếu giải pháp 2 không đủ
        # await asyncio.sleep(0.2)

        logger.info(f"Successfully cut segment to {output_path.name}")
        return True

    except Exception as e:
        # Log lỗi chi tiết hơn
        logger.error(f"Error cutting segment {start_time:.3f}s-{end_time:.3f}s into {output_path.name}: {e}", exc_info=True)
        return False

    finally:
        # Đảm bảo đóng clip đã được tạo để ghi
        if video_clip_to_write:
            try:
                # logger.debug(f"Closing clip for {output_path.name}")
                video_clip_to_write.close()
            except Exception as e_close:
                # Ghi log lỗi khi đóng clip nhưng không dừng chương trình
                logger.error(f"Error closing video clip for {output_path.name}: {e_close}")

async def concatenate_segments_v2(segment_paths: List[Path], output_path: Path) -> bool:
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
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Ghi file video tổng hợp
        await loop.run_in_executor(
            None,
            lambda: final_clip.write_videofile(
                str(output_path),
                codec="libx264", 
                audio_codec="aac",
                temp_audiofile=f"{str(output_path)}_temp_audio.m4a",
                remove_temp=True,
                logger=None,
                preset='medium', # Hoặc 'fast', 'faster'
                ffmpeg_params=['-crf', '23'] # Quality setting for libx264
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
                

async def concatenate_segments(segment_paths: List[Path], output_path: Path) -> bool:
    try:
        import cv2
        import numpy as np
        
        if not segment_paths:
            logger.error("No segment paths provided for concatenation.")
            return False
            
        logger.info(f"Concatenating {len(segment_paths)} segments into {output_path}")
        
        # Đọc video đầu tiên để lấy thông số
        first_video = cv2.VideoCapture(str(segment_paths[0]))
        if not first_video.isOpened():
            logger.error(f"Could not open first video: {segment_paths[0]}")
            return False
            
        # Lấy thông số từ video đầu tiên
        frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = first_video.get(cv2.CAP_PROP_FPS)
        first_video.release()
        
        # Khởi tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        # Xử lý từng video
        for video_path in segment_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}, skipping...")
                continue
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                    
            cap.release()
            
        # Đóng VideoWriter
        out.release()
        
        # Thêm audio vào video đã ghép
        # Sử dụng FFmpeg để copy audio từ các segments vào video cuối
        try:
            import ffmpeg
            
            # Tạo file danh sách input
            with open('concat_list.txt', 'w') as f:
                for path in segment_paths:
                    f.write(f"file '{path}'\n")
            
            # Sử dụng FFmpeg để ghép audio
            stream = ffmpeg.input('concat_list.txt', f='concat', safe=0)
            stream = ffmpeg.output(stream, str(output_path), c='copy', y=None)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            
        except Exception as e:
            logger.error(f"Error adding audio to concatenated video: {e}")
            return False
        finally:
            # Xóa file tạm
            if os.path.exists('concat_list.txt'):
                os.remove('concat_list.txt')
        
        logger.info(f"Successfully concatenated segments to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in concatenate_segments_v2: {e}", exc_info=True)
        return False