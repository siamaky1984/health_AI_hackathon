import time
from datetime import datetime
import os
import pyautogui
import sys

def record_streamlit(duration=90):
    """Record Streamlit interaction for 90 seconds"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"recordings/session_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Recording will start in 5 seconds...")
    print("Move your mouse to any corner of the screen to stop recording")
    time.sleep(5)
    
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    
    try:
        while time.time() < end_time:
            # Take screenshot
            screenshot = pyautogui.screenshot()
            
            # Save screenshot
            screenshot.save(f"{output_dir}/frame_{frame_count:04d}.png")
            
            # Update progress
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            print(f"\rRecording... {remaining:.0f} seconds remaining. Frames: {frame_count}", end="")
            
            frame_count += 1
            
            # Automated interactions
            current_time = elapsed
            
            # Data Input Tab (0-20 seconds)
            if 0 <= current_time < 20:
                if frame_count == 10:  # After 1 second
                    pyautogui.click(200, 200)  # Click first tab
                    time.sleep(0.5)
                    pyautogui.write('75')
            
            # Predictions Tab (20-40 seconds)
            elif 20 <= current_time < 40:
                if current_time == 20:
                    pyautogui.click(300, 200)  # Click second tab
            
            # Visualizations Tab (40-60 seconds)
            elif 40 <= current_time < 60:
                if current_time == 40:
                    pyautogui.click(400, 200)  # Click third tab
            
            # Health Recommendations Tab (60-75 seconds)
            elif 60 <= current_time < 75:
                if current_time == 60:
                    pyautogui.click(500, 200)  # Click fourth tab
            
            # Chat Assistant Tab (75-90 seconds)
            elif 75 <= current_time < 90:
                if current_time == 75:
                    pyautogui.click(600, 200)  # Click fifth tab
                    time.sleep(0.5)
                    pyautogui.click(400, 300)  # Click chat input
                    pyautogui.write("How can I improve my sleep?")
                    pyautogui.press('enter')
            
            time.sleep(0.1)  # 10 FPS
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        print(f"\nRecording completed!")
        print(f"Frames captured: {frame_count}")
        print(f"Output directory: {output_dir}")
        
        # Try to convert to video if ffmpeg is available
        try:
            os.system(f"ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_dir}_video.mp4")
            print(f"Video saved as: {output_dir}_video.mp4")
        except:
            print("Note: ffmpeg not found. Frames saved as individual images.")

def get_mouse_position():
    """Utility function to help find screen coordinates"""
    print("Move your mouse to the desired position...")
    time.sleep(3)
    x, y = pyautogui.position()
    print(f"Mouse position: x={x}, y={y}")

if __name__ == "__main__":
    # Create recordings directory if it doesn't exist
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    print("\nIMPORTANT SETUP:")
    print("1. Make sure your Streamlit app is running")
    print("2. Position the Streamlit window where you want it")
    print("3. Would you like to:")
    print("   a) Find mouse positions for tabs")
    print("   b) Start recording")
    
    choice = input("Enter 'a' or 'b': ").lower()
    
    if choice == 'a':
        while True:
            get_mouse_position()
            if input("Get another position? (y/n): ").lower() != 'y':
                break
    else:
        try:
            record_streamlit()
        except Exception as e:
            print(f"Error: {e}")
            print("Try running with 'a' first to get correct mouse positions")