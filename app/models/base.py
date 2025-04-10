# app/models/summarization.py
from pydantic import BaseModel
from typing import Optional, List

class TaskResponse(BaseModel):
    task_id: str
    message: str

class TaskStatusEnum:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    result_url: Optional[str] = None

# >> Model quan trọng cho việc này <<
class TimedWord(BaseModel):
    """Đại diện cho một từ với thông tin thời gian."""
    word: str
    start: float
    end: float

# >> Model quan trọng cho việc này <<
class Segment(BaseModel):
    """Đại diện cho một phân đoạn video/audio đã được phân chia."""
    id: int
    text: str             # Toàn bộ văn bản của segment
    start_time: float     # Thời điểm bắt đầu segment
    end_time: float       # Thời điểm kết thúc segment
    duration: float       # Thời lượng segment
    score: float = 0.0    # Điểm quan trọng (sẽ được tính sau)
    words: List[TimedWord] # Danh sách các từ TimedWord thuộc segment này