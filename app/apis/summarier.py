from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, Union, Dict, Any, List
import traceback
import os
import uuid
import asyncio
import shutil
from pathlib import Path
import time

from app.models.base import TaskResponse, TaskStatus, TaskStatusEnum
from app.config import get_config
from app.utils.pipeline import summary_video

# Get configuration
config = get_config()

# Dictionary to store task statuses
task_status_store = {}

router = APIRouter(
    prefix="/api/v1",
    tags=["Video Meeting Summarizer"],
)

async def process_video_task(
    task_id: str, 
    video_path: Path, 
    target_duration: int
):
    """Background task to process video summarization"""
    
    try:
        # Update task status to processing
        task_status_store[task_id] = {
            "status": TaskStatusEnum.PROCESSING,
            "message": "Started processing video",
            "current_step": "extracting_audio"
        }
        
        # Process the video
        summary_path = await summary_video(
            video_path=video_path,
            target_duration=target_duration
        )
        
        if not summary_path:
            raise ValueError("Failed to generate summary video")
        
        # Get relative path for URL
        relative_path = summary_path.relative_to(config.BASE_DIR.parent)
        result_url = f"/api/v1/video/{str(relative_path).replace(os.sep, '/')}"
        
        # Update task status to completed
        task_status_store[task_id] = {
            "status": TaskStatusEnum.COMPLETED,
            "message": "Summary generation completed",
            "result_url": result_url
        }
        
    except Exception as e:
        # Update task status to failed
        error_message = f"Error: {str(e)}"
        task_status_store[task_id] = {
            "status": TaskStatusEnum.FAILED,
            "message": error_message
        }
        print(f"Task {task_id} failed: {error_message}")
        traceback.print_exc()

@router.post("/summarize")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_duration: int = Form(300),  # Default 5 minutes (300 seconds)
):
    """
    Upload a video file and start the summarization process.
    Returns a task ID to check the status.
    """
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create a temporary directory for this task if needed
        temp_video_dir = config.video_upload_path
        os.makedirs(temp_video_dir, exist_ok=True)
        
        # Save the uploaded file
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1]
        video_filename = f"upload_{task_id}{file_extension}"
        video_path = temp_video_dir / video_filename
        
        # Save the uploaded file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Set initial task status
        task_status_store[task_id] = {
            "status": TaskStatusEnum.PENDING,
            "message": "Task queued for processing"
        }
        
        # Start the background task
        background_tasks.add_task(
            process_video_task,
            task_id=task_id,
            video_path=video_path,
            target_duration=target_duration
        )
        
        return TaskResponse(
            task_id=task_id,
            message="Video upload successful. Processing started."
        )
        
    except Exception as e:
        error_message = f"Error processing upload: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a video processing task.
    """
    if task_id not in task_status_store:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task_info = task_status_store[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task_info.get("status", TaskStatusEnum.PENDING),
        message=task_info.get("message", ""),
        result_url=task_info.get("result_url", None)
    )

@router.get("/video/{path:path}")
async def get_video_file(path: str):
    """
    Serve the generated video files.
    """
    full_path = config.BASE_DIR.parent / path
    
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=full_path,
        media_type="video/mp4",
        filename=os.path.basename(path)
    )
