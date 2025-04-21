# filepath: d:\Sgroup\Sgroup-AI\video-meet-summarier\app\utils\skim_generator.py
import math
from typing import List, Dict
from pathlib import Path
import logging
import asyncio # Để gọi hàm async khác

from app.models.base import Segment
from app.config import get_config
from app.utils.video_processor import cut_segment_refactored, concatenate_segments_ffmpeg
     
# Set up logger
logger = logging.getLogger(__name__)

config = get_config()

TEMP_DIR = config.TEMP_DIR
SUMMARY_DIR = config.SUMMARY_DIR
TEMP_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

async def generate_skim(
    segments: List[Segment],       # Danh sách segment đã có điểm
    target_duration: int,        # Thời lượng mong muốn (giây)
    original_video_path: Path, # Đường dẫn video gốc
    output_filename_base: str    # Tên file output (không có đuôi)
) -> Path:
    """Chọn lọc segment theo thuật toán Greedy Knapsack và tạo video tóm tắt."""
    if not segments:
        raise ValueError("No segments provided to generate skim.")
    if target_duration <= 0:
        raise ValueError("Target duration must be positive.")

    # --- Bước 9: Lựa chọn Đoạn & Ghép nối ---
    # 1. Tính hiệu quả (efficiency) và lọc segment không hợp lệ
    segment_efficiencies = []
    total_original_duration = 0.0
    for seg in segments:
        total_original_duration += seg.duration
        # Bỏ qua segment quá ngắn hoặc duration=0 hoặc score=0 (không có giá trị)
        if seg.duration > 0.1 and seg.score > 1e-9:
            efficiency = seg.score / seg.duration
            segment_efficiencies.append({'segment': seg, 'efficiency': efficiency})
        else:
            logger.info(f"Skipping segment {seg.id} due to short/zero duration ({seg.duration:.2f}s) or zero score ({seg.score:.4f}).")

    if not segment_efficiencies:
        raise ValueError("No valid segments found after filtering for efficiency calculation.")
    logger.info(f"Calculated efficiency for {len(segment_efficiencies)} valid segments (Total original duration: {total_original_duration:.2f}s).")

    # 2. Sắp xếp theo hiệu quả giảm dần
    segment_efficiencies.sort(key=lambda x: x['efficiency'], reverse=True)

    # 3. Thuật toán Tham lam (Greedy Knapsack) để chọn segment
    selected_segments_info: List[Dict] = [] # Lưu trữ {'segment': Segment, 'efficiency': float}
    current_total_duration = 0.0
    remaining_time = float(target_duration)
    # Dùng set để theo dõi chỉ số của các segment đã chọn trong list gốc (segment_efficiencies)
    used_indices = set()

    # Vòng lặp chính: Chọn segment hiệu quả nhất phù hợp
    for i, item in enumerate(segment_efficiencies):
        segment = item['segment']
        if remaining_time <= 0: break # Đã đủ hoặc vượt thời lượng

        if segment.duration <= remaining_time:
            selected_segments_info.append(item)
            remaining_time -= segment.duration
            current_total_duration += segment.duration
            used_indices.add(i)
            logger.debug(f"Selected segment {segment.id} (eff: {item['efficiency']:.4f}, dur: {segment.duration:.2f}s). Remaining time: {remaining_time:.2f}s")

    logger.info(f"Greedy selection phase done. Selected {len(selected_segments_info)} segments. Current duration: {current_total_duration:.2f}s. Remaining time: {remaining_time:.2f}s")

    # 4. Cải tiến: Tìm segment nhỏ hơn để lấp đầy thời gian còn lại
    # Ngưỡng nhỏ để xem xét lấp đầy (ví dụ > 0.5 giây)
    fill_threshold = 0.5
    fill_threshold_percent = 0.5  # 0.5% của thời lượng mục tiêu
    # fill_threshold = max(0.5, target_duration * fill_threshold_percent / 100)  
    logger.info(f"Fill threshold: {fill_threshold:.2f}s ({fill_threshold_percent:.2f}% of target duration)")
    if remaining_time > fill_threshold:
        logger.info(f"Attempting to fill remaining time: {remaining_time:.2f}s")
        best_fit_segment_info = None
        best_fit_index = -1

        # Tìm trong các segment *chưa được sử dụng*
        for i, item in enumerate(segment_efficiencies):
            if i not in used_indices:
                segment = item['segment']
                # Chọn segment đầu tiên tìm thấy mà vừa vặn
                if segment.duration <= remaining_time:
                    best_fit_segment_info = item
                    best_fit_index = i
                    logger.debug(f"Found potential fill segment {segment.id} (dur: {segment.duration:.2f}s)")
                    break # Lấy cái đầu tiên

        if best_fit_segment_info:
            segment = best_fit_segment_info['segment']
            selected_segments_info.append(best_fit_segment_info)
            remaining_time -= segment.duration
            current_total_duration += segment.duration
            used_indices.add(best_fit_index)
            logger.info(f"Filled remaining time with segment {segment.id} (dur: {segment.duration:.2f}s).")

    if not selected_segments_info:
        raise ValueError("No segments selected for the summary. Check target duration or segment scores/durations.")

    # Lấy danh sách các đối tượng Segment đã chọn
    final_selected_segments = [info['segment'] for info in selected_segments_info]

    # 5. Sắp xếp lại các segment đã chọn theo thời gian gốc để ghép nối đúng thứ tự
    final_selected_segments.sort(key=lambda s: s.start_time)
    logger.info(f"Final selection: {len(final_selected_segments)} segments. Final duration: {current_total_duration:.2f}s.")
    logger.info("Selected segments:")
    for seg in final_selected_segments:
        logger.info(f"  - Segment {seg.id}: {seg.start_time:.2f}s to {seg.end_time:.2f}s (dur: {seg.duration:.2f}s, score: {seg.score:.4f})")
    
    # kiểm tra xem có seg nào final_selected_segments không hợp lệ không hay trùng lặp
    for i, seg in enumerate(final_selected_segments):
        if seg.start_time < 0 or seg.end_time <= seg.start_time or seg.duration <= 0:
            logger.error(f"Invalid segment {seg.id}: start_time={seg.start_time}, end_time={seg.end_time}, duration={seg.duration}.")
            raise ValueError(f"Invalid segment {seg.id}: start_time={seg.start_time}, end_time={seg.end_time}, duration={seg.duration}.")
        if i > 0 and final_selected_segments[i-1].end_time > seg.start_time:
            logger.error(f"Overlapping segments detected: {final_selected_segments[i-1].id} and {seg.id}.")
            raise ValueError(f"Overlapping segments detected: {final_selected_segments[i-1].id} and {seg.id}.")
    
    # xóa các trùng lặp trong final_selected_segments nếu có
    unique_segments = []
    seen_ids = set()
    for seg in final_selected_segments:
        if seg.id not in seen_ids:
            unique_segments.append(seg)
            seen_ids.add(seg.id)
    final_selected_segments = unique_segments
    logger.info(f"After removing duplicates, {len(final_selected_segments)} unique segments remain.")

    # 6. Cắt các đoạn video tuần tự
    segment_file_paths: List[Path] = [] # Lưu đường dẫn file tạm của các segment đã cắt
    results = [] # Lưu kết quả hoặc exception
    output_file_suffix = ".mp4" # Hoặc lấy từ video gốc nếu muốn
    
    logger.info(f"Starting to cut {len(final_selected_segments)} segments sequentially...")
    
    for i, segment in enumerate(final_selected_segments):
        # Tạo tên file tạm duy nhất cho mỗi segment
        segment_temp_path = TEMP_DIR / f"{output_filename_base}_temp_seg_{segment.id}{output_file_suffix}"
        segment_file_paths.append(segment_temp_path)
        
        logger.info(f"--- Cutting segment {segment.id} ---")
        try:
            # Gọi hàm cắt trực tiếp, không tạo danh sách các coroutine trước
            result = await cut_segment_refactored(
                original_video_path, 
                segment.start_time, 
                segment.end_time, 
                segment_temp_path
            )
            
            results.append(result) # Thêm kết quả (True/False) vào list
            if result is True:
                logger.info(f"--- Successfully cut segment {segment.id} ---")
            else:
                logger.error(f"--- Failed to cut segment {segment.id} (path: {segment_temp_path}). Function returned False. ---")
        except Exception as e:
            logger.error(f"--- Failed to cut segment {segment.id} (path: {segment_temp_path}): {e} ---", exc_info=True)
            results.append(e) # Thêm exception vào list

    # Lọc ra các đường dẫn của những segment cắt thành công
    successful_cut_paths = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to cut segment {final_selected_segments[i].id} (path: {segment_file_paths[i]}): {result}")
        elif result is True: # Giả sử cut_segment trả về True khi thành công
            successful_cut_paths.append(segment_file_paths[i])
        else: # cut_segment trả về False
            logger.error(f"Failed to cut segment {final_selected_segments[i].id} (path: {segment_file_paths[i]}). Function returned False.")

    if not successful_cut_paths:
        # Dọn dẹp các file tạm đã tạo (nếu có) trước khi raise lỗi
        for temp_path in segment_file_paths:
             temp_path.unlink(missing_ok=True)
        raise RuntimeError("Failed to cut any valid video segments.")
    logger.info(f"Successfully cut {len(successful_cut_paths)} segments.")

    # 7. Ghép nối các đoạn đã cắt thành công
    final_summary_path = SUMMARY_DIR / f"{output_filename_base}{output_file_suffix}"
    logger.info(f"Concatenating {len(successful_cut_paths)} segments into {final_summary_path}...")
    # Giả định concatenate_segments là async
    # concatenation_success = await concatenate_segments(successful_cut_paths, final_summary_path)
    concatenation_success = await concatenate_segments_ffmpeg(successful_cut_paths, final_summary_path)

    # 8. Dọn dẹp file segment tạm sau khi ghép nối (bất kể thành công hay không)
    logger.info("Cleaning up temporary segment files...")
    delete_tasks = []
    for temp_path in segment_file_paths: # Xóa tất cả các file tạm đã được tạo đường dẫn
        if temp_path.exists(): # Chỉ xóa nếu file thực sự tồn tại
             # Tạo coroutine xóa (ví dụ, nếu dùng aiofiles)
             # Hoặc xóa đồng bộ nếu không quá nhiều file
             try:
                temp_path.unlink()
             except Exception as e:
                logger.error(f"Could not delete temporary segment file {temp_path}: {e}")
    #     Nếu dùng async delete: delete_tasks.append(aiofiles.os.remove(str(temp_path)))
    # Nếu dùng async delete: await asyncio.gather(*delete_tasks, return_exceptions=True)
    
    return final_summary_path