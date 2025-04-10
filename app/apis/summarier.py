from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Union, Dict, Any
import traceback

import numpy as np

router = APIRouter(
    prefix="/api/v1",
    tags=["Video Meeting Summarizer"],
)

@router.post("/summarize")
async def process(
    file: UploadFile = File(...),
    summary_type: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    pass
