import cv2
import os
import asyncio
import requests
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from typing import List

CACHE_DIR = "cache"
# Initialize CLIP Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Download video from URL
async def download_video(url: str, filename: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download video from {url}")

# Function to process each chunk of frames
def process_chunk(video_path, chunk_start, chunk_end, output_dir, chunk_id, threshold=70000):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)  # Start processing from the given chunk start frame
    
    prev_frame = None
    saved_frames = 0
    first_frame = None
    last_frame_path = None  # Path to the last frame saved

    for i in range(chunk_start, chunk_end):
        ret, frame = cap.read()
        if not ret:
            break  # Break if we can't read more frames
        
        # Compare with previous frame using L2 norm
        if prev_frame is None or cv2.norm(prev_frame, frame, cv2.NORM_L2) > threshold:
            # Save frame as PNG with chunk and index in filename
            frame_filename = os.path.join(output_dir, f"frame_{chunk_id}_{saved_frames+1}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
            if first_frame is None:
                first_frame = frame  # Save the first frame
            last_frame_path = frame_filename  # Continuously update last frame path

        prev_frame = frame

    cap.release()
    return first_frame, last_frame_path

# Function to compare boundary frames between chunks and remove similar images
def compare_boundary_frames(boundary_frames, threshold=70000):
    for i in range(len(boundary_frames) - 1):
        last_frame_path = boundary_frames[i][1]  # Path of the last frame of the current chunk
        first_frame = boundary_frames[i + 1][0]  # First frame (NumPy array) of the next chunk

        if last_frame_path is not None and first_frame is not None:
            # Load the last frame of the current chunk
            last_frame = cv2.imread(last_frame_path)

            # Calculate the L2 norm
            similarity = cv2.norm(last_frame, first_frame, cv2.NORM_L2)

            if similarity < threshold:
                os.remove(last_frame_path)  # Remove the last frame of the current chunk
            else:
                continue

# Async function to process chunks
async def extract_key_frames_async(video_path: str, output_dir="key_frames", num_chunks=10):
    # Initialize the video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Determine chunk size and boundaries
    chunk_size = frame_count // num_chunks
    chunk_ranges = [
        (i * chunk_size, (i + 1) * chunk_size if i < num_chunks - 1 else frame_count)
        for i in range(num_chunks)
    ]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use ThreadPoolExecutor for CPU-bound operations
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = []
        for idx, (start, end) in enumerate(chunk_ranges):
            tasks.append(loop.run_in_executor(
                executor, process_chunk, video_path, start, end, output_dir, idx
            ))
        
        # Wait for all tasks to complete
        boundary_frames = await asyncio.gather(*tasks)

    # Compare boundary frames between chunks and remove similar images
    compare_boundary_frames(boundary_frames)

# Create a GIF from saved PNG frames
def create_gif_from_frames(png_files, output_gif="output.gif"):
    images = [Image.open(png_file) for png_file in png_files]
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=500, loop=0)

# Process a single frame with CLIP asynchronously
async def process_frame_with_clip(frame, keywords):
    # Convert frame (NumPy array) to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
   
    # Check if only 1 keyword is provided
    if len(keywords) == 1:
        keywords = [keywords[0], f"99"]  # Duplicate the keyword to match with text and image
    # Process text and image for CLIP model
    inputs = processor(text=keywords, images=frame_pil, return_tensors="pt", padding=True)
    
    # Get CLIP model outputs
    outputs = model(**inputs)
    
    # Get similarity scores
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # Apply softmax to get label probabilities
    return probs

# Process video: download, extract key frames, and generate GIF
async def process_video(video_url: str, keywords: List[str], gif_filename: str):
    # Ensure cache directory exists
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # Download video
    video_filename = os.path.join(CACHE_DIR, f"{video_url.split('/')[-1]}.mp4")
    await download_video(video_url, video_filename)

    # Create output directory for PNG frames
    png_output_dir = os.path.join(CACHE_DIR, "key_frames")
    os.makedirs(png_output_dir, exist_ok=True)

    # Extract key frames
    await extract_key_frames_async(video_filename, output_dir=png_output_dir)

    # Collect all PNG files in the output directory
    png_files = sorted(
        [os.path.join(png_output_dir, f) for f in os.listdir(png_output_dir) if f.endswith(".png")]
    )

    # Ensure PNG files are found before creating GIF
    if not png_files:
        raise Exception("No PNG files found to create GIF. Extraction might have failed.")

    # Initialize list to hold frames that match the keywords
    matching_frames = []

    # If keywords is empty, skip CLIP processing and directly add all frames to matching_frames
    if not keywords:
        matching_frames = png_files
    else:
        # Process each PNG file asynchronously
        tasks = []
        for png_file in png_files:
            # Load frame
            frame = cv2.imread(png_file)
            
            # Create async task for processing the frame
            tasks.append(process_frame_with_clip(frame, keywords))

        # Await all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Process the results and filter frames that match the keywords
        for i, probs in enumerate(results):
            if len(keywords) == 1:
                if probs.argmax() == 1:
                    continue
            if probs.max() > 0.7:  # Threshold to decide if the frame matches the keywords
                matching_frames.append(png_files[i])

    # Ensure matching frames are found before creating GIF
    if not matching_frames:
        raise Exception("No frames matched the keywords. Please try with different keywords.")
    
    # Create GIF
    create_gif_from_frames(matching_frames, gif_filename)

    # Clean up PNG files after creating the GIF
    for frame in png_files:
        os.remove(frame)

    # Optionally, remove the `key_frames` directory itself
    os.rmdir(png_output_dir) 