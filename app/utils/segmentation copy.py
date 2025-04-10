from typing import List, Tuple
import logging
from app.models.base import TimedWord, Segment
from app.config import get_config

logger = logging.getLogger(__name__)

config_settings = get_config()

def calc_pauses(transcripts: List[TimedWord]) -> List[Tuple[float, int]]:
    pauses = []
    if len(transcripts) < 2:
        return pauses
    for index, word in enumerate(transcripts):
        if index > 0:
            duration = word.start - transcripts[index - 1].end
            if duration > 0:
                pauses.append((duration, index - 1))
    return pauses

async def segment_transcript(transcript: List[TimedWord]) -> List[Segment]:
    if not transcript:
        print("Input transcript list is empty, returning empty segments.")
        return []
    
    # print ra 5 timedword đầu tiên
    print("First 5 timed words:")
    
    for word in transcript[:5]:
        print(word)
    
    pauses = calc_pauses(transcript)
    if not pauses:
        segment_text = " ".join([w.word for w in transcript])
        start_time = transcript[0].start if transcript else 0.0
        end_time = transcript[-1].end if transcript else 0.0
        duration = max(0.0, end_time - start_time)
        print("No pauses found between words, creating a single segment.")
        return [Segment(id=0, text=segment_text, start_time=start_time, end_time=end_time, duration=duration, words=transcript)]
    
    segment_boundaries = [0]
    n = config_settings.SEGMENTATION_N
    m = config_settings.SEGMENTATION_M
    
    window_size = 2 * m + 1 
    
    for i in range(len(pauses)):
        current_pause_duration, current_pause_idx = pauses[i]
        
        sstart_idx = max(0, i - m)
        send_idx = min(len(pauses), i + m + 1)
        window = pauses[sstart_idx:send_idx]
        
        if not window:
            print(f"Window is empty at index {i}, skipping.")
            continue
        
        window_sorted = sorted(window, key=lambda x: x[0], reverse=True)
        longest_pause = window_sorted[0][0]
        longest_pause_index = window_sorted[0][1]
        
        # is_checked = (longest_pause_index == i) 
        is_checked = abs(current_pause_duration - longest_pause) < 1e-9
        
        if is_checked:
            second_longest_pause = 0.0
            
            if len(window_sorted) > 1:
                second_longest_pause = window_sorted[1][0]
                
            is_checked_2 = False
            threshold_pause = 1e-9
            if second_longest_pause > threshold_pause:
                is_checked_2 = (current_pause_duration >= n * second_longest_pause)
            elif current_pause_duration > 0.1:
                is_checked_2 = True

            if is_checked_2:
                current_idx =  current_pause_idx  + 1
                if current_idx != segment_boundaries[-1]:
                    segment_boundaries.append(current_idx)
                    print(f"Adding segment boundary at index {current_idx} with pause duration {current_pause_duration}.")
    
    if len(transcript) not in segment_boundaries:
        segment_boundaries.append(len(transcript))
        print(f"Adding final segment boundary at index {len(transcript)}.")
    
    segment_boundaries = sorted(list(set(segment_boundaries)))
    
    segments: List[Segment] = []
    counter = 0
    
    for i in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i + 1]
        
        if start_idx >= end_idx:
            continue
        
        segment_handle = transcript[start_idx:end_idx]
        if not segment_handle:
            print(f"Segment handle is empty for indices {start_idx} to {end_idx}, skipping.")
            continue
        
        segment_text = " ".join([w.word for w in segment_handle])
        if not segment_text.strip():
            print(f"Segment {counter} is empty, skipping.")
            continue
        
        start_time = segment_handle[0].start if segment_handle else 0.0
        end_time = segment_handle[-1].end if segment_handle else 0.0
        
        duration = max(0.0, end_time - start_time)
        
        if duration <= 0:
            print(f"Segment {counter} has non-positive duration, skipping.")
            continue
        
        segments.append(Segment(
            id=counter,
            text=segment_text,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            words=segment_handle
        ))
        counter += 1
        print(f"Segment {counter} created with duration {duration:.3f}s.")
    
    return segments
