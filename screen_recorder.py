import numpy as np
import cv2
import time
from datetime import datetime
import os
from PIL import Image, ImageGrab

def record_screen():
    # Create output directory
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    # Set up file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"recordings/screen_{timestamp}"
    os.makedirs(output_path, exist_ok=True)
    
    print("Recording will start in 3 seconds...")
    print("Press Ctrl+C to stop recording")
    time.sleep(3)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture screenshot
            screenshot = ImageGrab.grab()
            
            # Save screenshot
            screenshot.save(f"{output_path}/frame_{frame_count:04d}.png")
            
            # Update counter and display progress
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"\rRecording... Frames: {frame_count}, Time: {elapsed_time:.1f}s, FPS: {fps:.1f}", end="")
            
            # Control frame rate
            time.sleep(0.1)  # Approximately 10 FPS
            
    except KeyboardInterrupt:
        print("\nStopping recording...")
    
    finally:
        print(f"\nRecording saved in: {output_path}")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {time.time() - start_time:.1f} seconds")
        
        # Convert frames to video if ffmpeg is available
        try:
            import subprocess
            video_file = f"recordings/recording_{timestamp}.mp4"
            subprocess.run([
                'ffmpeg',
                '-framerate', '10',
                '-i', f"{output_path}/frame_%04d.png",
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                video_file
            ])
            print(f"Video saved as: {video_file}")
        except:
            print("Note: ffmpeg not found. Frames saved as individual images.")

if __name__ == "__main__":
    # Verify imports
    print("Checking required packages...")
    try:
        import PIL
        print("PIL version:", PIL.__version__)
        print("All packages loaded successfully!")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please run: conda install pillow")
        exit(1)
    
    # Start recording
    record_screen()