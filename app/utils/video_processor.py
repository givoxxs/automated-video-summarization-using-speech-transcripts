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
        return [] # Trả về list rỗng khi có lỗ

async def cut_segment(video_path: Path, start_time: float, end_time: float, output_path: Path) -> bool:
    """
    Cắt một phân đoạn từ video gốc dựa trên thời gian bắt đầu và kết thúc.
    Phiên bản cải tiến với xử lý lỗi tốt hơn và các biện pháp đảm bảo tính toàn vẹn của video.
    
    Args:
        video_path: Đường dẫn đến file video gốc
        start_time: Thời gian bắt đầu (giây)
        end_time: Thời gian kết thúc (giây)
        output_path: Đường dẫn file đầu ra
        
    Returns:
        bool: True nếu cắt thành công, False nếu thất bại
    """
    video_clip = None
    video_clip_to_write = None
    loop = asyncio.get_event_loop()

    # Kiểm tra đầu vào
    if start_time >= end_time:
        logger.error(f"Invalid time range: start_time {start_time:.3f} >= end_time {end_time:.3f}")
        return False

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False

    try:
        # Log thời gian với độ chính xác cao hơn
        logger.info(f"Cutting segment from {start_time:.3f}s to {end_time:.3f}s -> {output_path.name}")

        # Đảm bảo thư mục chứa output_path tồn tại
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # --- Đọc thông tin video và điều chỉnh thời gian nếu cần ---
        adjusted_start_time = max(0, start_time)  # Đảm bảo start_time không âm
        adjusted_end_time = end_time
        
        # Đọc video clip gốc để lấy thông tin
        video_info = await loop.run_in_executor(
            None, 
            lambda: VideoFileClip(str(video_path), target_resolution=None)
        )
        
        try:
            # Lấy thời lượng và điều chỉnh end_time nếu vượt quá
            duration = video_info.duration
            if adjusted_end_time > duration:
                logger.warning(f"End time {adjusted_end_time:.3f}s exceeds video duration {duration:.3f}s, adjusting to {duration:.3f}s")
                adjusted_end_time = duration
            
            # Đảm bảo start_time < end_time sau khi điều chỉnh
            if adjusted_start_time >= adjusted_end_time:
                logger.error(f"Invalid adjusted time range: {adjusted_start_time:.3f}s >= {adjusted_end_time:.3f}s")
                video_info.close()
                return False
                
            # Đóng video_info sau khi lấy thông tin cần thiết
            video_info.close()
        except Exception as e:
            logger.warning(f"Error accessing video duration: {e}, proceeding with original times")
            try:
                video_info.close()
            except:
                pass
                
        # --- Cắt video với thời gian đã điều chỉnh ---
        logger.info(f"Creating subclip from {adjusted_start_time:.3f}s to {adjusted_end_time:.3f}s")
        
        # Tạo clip mới để cắt (tránh vấn đề state với clip trước đó)
        video_clip = await loop.run_in_executor(
            None,
            lambda: VideoFileClip(str(video_path), target_resolution=None)
        )
        
        # Thực hiện cắt video
        video_clip_to_write = video_clip.subclipped(adjusted_start_time, adjusted_end_time)
        
        # Kiểm tra xem subclip có thành công không
        if not video_clip_to_write:
            raise ValueError(f"Subclip operation failed for {adjusted_start_time:.3f}s-{adjusted_end_time:.3f}s")
        
        # Kiểm tra clip có audio không và thời lượng
        has_audio = video_clip_to_write.audio is not None
        clip_duration = video_clip_to_write.duration
        
        logger.info(f"Subclip created successfully. Duration: {clip_duration:.3f}s, Has audio: {has_audio}")
        
        # Đóng video_clip gốc sau khi đã tạo subclip
        video_clip.close()
        video_clip = None  # Đánh dấu đã đóng

        # Ghi file video đã cắt với cài đặt tối ưu
        await loop.run_in_executor(
            None,
            lambda: video_clip_to_write.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac" if has_audio else None,  # Chỉ encode audio nếu có
                temp_audiofile=f"{str(output_path)}_temp_audio.m4a" if has_audio else None,
                remove_temp=True,
                logger=None,  # Giữ logger=None để tránh spam progress bar
                preset='fast',  # Sử dụng preset nhanh hơn
                threads=2,  # Đa luồng cho tốc độ nhanh
                ffmpeg_params=['-crf', '23']  # Cân bằng giữa chất lượng và kích thước
            )
        )

        # Kiểm tra xem file đầu ra có tồn tại không
        if not output_path.exists():
            raise FileNotFoundError(f"Output file {output_path} was not created")

        logger.info(f"Successfully cut segment to {output_path.name}")
        return True

    except Exception as e:
        # Log lỗi chi tiết hơn
        logger.error(f"Error cutting segment {start_time:.3f}s-{end_time:.3f}s into {output_path.name}: {e}", exc_info=True)
        
        # Xóa file đầu ra nếu tồn tại nhưng có lỗi
        if output_path.exists():
            try:
                output_path.unlink()
                logger.info(f"Deleted incomplete output file: {output_path}")
            except Exception as e_unlink:
                logger.error(f"Failed to delete incomplete output file: {e_unlink}")
                
        return False

    finally:
        # Đảm bảo đóng tất cả các clip để giải phóng tài nguyên
        for clip in [video_clip, video_clip_to_write]:
            if clip:
                try:
                    clip.close()
                except Exception as e_close:
                    logger.error(f"Error closing video clip: {e_close}")

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
    """
    Ghép nối nhiều phân đoạn video thành một video tổng hợp sử dụng MoviePy.
    Phiên bản cải tiến với kiểm tra lỗi tốt hơn và sử dụng MoviePy để giữ cả video và audio.
    
    Args:
        segment_paths: Danh sách các đường dẫn đến file video phân đoạn
        output_path: Đường dẫn file đầu ra tổng hợp
        
    Returns:
        bool: True nếu ghép nối thành công, False nếu thất bại
    """
    clips = []
    final_clip = None
    loop = asyncio.get_event_loop()
    
    try:
        # Kiểm tra đầu vào
        if not segment_paths:
            logger.error("No segment paths provided for concatenation.")
            return False
            
        # Lọc ra các file segment tồn tại
        valid_segment_paths = []
        for path in segment_paths:
            if not os.path.exists(path):
                logger.warning(f"Segment file not found: {path}, skipping...")
                continue
            if os.path.getsize(path) == 0:
                logger.warning(f"Segment file is empty: {path}, skipping...")
                continue
            valid_segment_paths.append(path)
            
        if not valid_segment_paths:
            logger.error("No valid segment files found for concatenation.")
            return False
            
        logger.info(f"Concatenating {len(valid_segment_paths)} valid segments into {output_path}")
        
        # Đảm bảo thư mục chứa output_path tồn tại
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Tải các clip và kiểm tra sự tương thích
        for i, path in enumerate(valid_segment_paths):
            try:
                logger.info(f"Loading segment {i+1}/{len(valid_segment_paths)}: {path.name}")
                
                # Tải VideoFileClip trong executor để không block event loop
                clip = await loop.run_in_executor(None, lambda p=path: VideoFileClip(str(p)))
                
                # Kiểm tra độ dài và thuộc tính clip
                if clip.duration < 0.1:
                    logger.warning(f"Segment {path.name} is too short ({clip.duration:.3f}s), skipping...")
                    clip.close()
                    continue
                    
                clips.append(clip)
                logger.info(f"Successfully loaded segment {path.name} (duration: {clip.duration:.3f}s)")
                
            except Exception as e:
                logger.error(f"Error loading segment {path.name}: {e}")
                # Tiếp tục với segment tiếp theo nếu có lỗi với segment này
        
        if not clips:
            logger.error("No valid video clips could be loaded for concatenation.")
            return False
            
        # Hiển thị tóm tắt các clip sẽ được ghép nối
        total_duration = sum(clip.duration for clip in clips)
        logger.info(f"Preparing to concatenate {len(clips)} clips with total duration of {total_duration:.3f}s")
        
        # Ghép nối các clip
        logger.info("Performing clip concatenation...")
        final_clip = await loop.run_in_executor(None, lambda: concatenate_videoclips(clips, method="compose"))
        
        if not final_clip:
            raise ValueError("Failed to concatenate video clips")
            
        # Ghi file video tổng hợp với cài đặt tối ưu
        logger.info(f"Writing final concatenated video to {output_path}")
        
        # Kiểm tra nếu final clip có audio
        has_audio = final_clip.audio is not None
        
        await loop.run_in_executor(
            None,
            lambda: final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac" if has_audio else None,
                temp_audiofile=f"{str(output_path)}_temp_audio.m4a" if has_audio else None,
                remove_temp=True,
                logger=None,
                preset='fast',
                threads=2,
                ffmpeg_params=['-crf', '23']
            )
        )
        
        # Kiểm tra file đầu ra
        if not output_path.exists():
            raise FileNotFoundError(f"Output file {output_path} was not created")
            
        logger.info(f"Successfully concatenated {len(clips)} segments to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error concatenating segments: {e}", exc_info=True)
        
        # Xóa file đầu ra nếu tồn tại nhưng có lỗi
        if output_path.exists():
            try:
                output_path.unlink()
                logger.info(f"Deleted incomplete output file: {output_path}")
            except Exception as e_unlink:
                logger.error(f"Failed to delete incomplete output file: {e_unlink}")
                
        return False
        
    finally:
        # Đóng tất cả các clips để giải phóng tài nguyên
        for clip in clips:
            if clip:
                try:
                    clip.close()
                except Exception as e:
                    logger.error(f"Error closing video clip: {e}")
                    
        if final_clip and final_clip not in clips:
            try:
                final_clip.close()
            except Exception as e:
                logger.error(f"Error closing final video clip: {e}")