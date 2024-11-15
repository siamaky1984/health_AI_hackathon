import os
import time
from datetime import datetime
import subprocess

def record_streamlit(duration=90):
    """Record screen for 90 seconds using macOS screencapture"""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"recordings/session_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nPreparing to record...")
    print(f"Duration: {duration} seconds")
    print("Recording will start in 5 seconds")
    print("Switch to your Streamlit window now!")
    
    # Countdown
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    
    try:
        while time.time() < end_time:
            # Capture screenshot using macOS screencapture
            frame_file = f"{output_dir}/frame_{frame_count:04d}.png"
            subprocess.run(['screencapture', '-x', frame_file])
            
            # Update progress
            frame_count += 1
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            print(f"\rRecording... {remaining:.0f} seconds remaining. Frames: {frame_count}", end="")
            
            # Wait for 0.1 seconds (10 frames per second)
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    
    finally:
        print("\n\nRecording completed!")
        print(f"Frames captured: {frame_count}")
        print(f"Output directory: {output_dir}")
        
        # Try to create video using ffmpeg
        try:
            video_file = f"recordings/streamlit_recording_{timestamp}.mp4"
            subprocess.run([
                'ffmpeg',
                '-framerate', '10',
                '-i', f"{output_dir}/frame_%04d.png",
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                video_file
            ])
            print(f"\nVideo saved as: {video_file}")
            
            # Clean up individual frames
            subprocess.run(['rm', '-rf', output_dir])
            print("Cleaned up individual frames")
            
        except Exception as e:
            print(f"\nNote: Couldn't create video (ffmpeg might not be installed)")
            print("Frames saved as individual images in:", output_dir)

if __name__ == "__main__":
    # Create recordings directory if it doesn't exist
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    print("STREAMLIT SCREEN RECORDER")
    print("========================")
    print("1. Make sure your Streamlit app is running")
    print("2. Position the Streamlit window where you want it")
    print("3. The recording will last 90 seconds")
    print("\nPress Enter when ready to start, or Ctrl+C to cancel...")
    
    try:
        input()
        record_streamlit()
    except KeyboardInterrupt:
        print("\nRecording cancelled")