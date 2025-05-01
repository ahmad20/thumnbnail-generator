from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from app.utils import process_video
from typing import List

app = FastAPI()

class VideoRequest(BaseModel):
    url: str
    keywords: List[str]

CACHE_DIR = "cache"

async def cleanup_cache():
    # Clean all files in the cache directory after sending the response
    for file in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Optionally, remove the cache directory itself
    if os.path.isdir(CACHE_DIR):
        os.rmdir(CACHE_DIR)

@app.post("/generate_thumbnail/")
async def generate_thumbnail(request: VideoRequest, background_tasks: BackgroundTasks):
    video_url = request.url
    keywords = request.keywords
    # Check if the GIF already exists in the cache
    gif_filename = f"{CACHE_DIR}/{video_url.split('/')[-1]}.gif".replace(".mp4", "")
    if os.path.exists(gif_filename):
        # Schedule cleanup after sending the response
        background_tasks.add_task(cleanup_cache)
        return FileResponse(gif_filename)
    
    # Process the video: download, extract key frames, and create GIF
    try:
        await process_video(video_url, keywords, gif_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")

    # Schedule cleanup after sending the response
    background_tasks.add_task(cleanup_cache)

    return FileResponse(gif_filename)  
