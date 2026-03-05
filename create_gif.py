
import cv2
from PIL import Image
import os

input_path = "download.mp4"
output_gif = "demo_preview.gif"

def create_gif():
    if not os.path.exists(input_path):
        print("Video not found.")
        return

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    frame_count = 0
    skip_rate = 2 # Process every 2nd frame to reduce size
    
    # Start from second 1 to skip initialization jitter
    start_frame = int(fps * 1)
    # End at second 6 (5 seconds duration)
    end_frame = int(fps * 6)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    
    print("Reading frames for GIF...")
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % skip_rate == 0:
            # Resize for README (width 600)
            height, width = frame.shape[:2]
            new_width = 600
            new_height = int(height * (new_width / width))
            resized = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            frames.append(pil_img)
            
        frame_count += 1
        current_frame += 1

    cap.release()
    
    if frames:
        print(f"Saving GIF with {len(frames)} frames...")
        # Duration is in milliseconds per frame. 
        # Original FPS / skip_rate -> new FPS. 
        # Duration = 1000 / (fps / skip_rate)
        duration_ms = 1000 / (fps / skip_rate)
        
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0
        )
        print(f"GIF saved: {output_gif}")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    create_gif()
