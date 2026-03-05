
import cv2
import os

input_path = "/Users/ajaykasu/Downloads/earthcam-inference/YTDowncom_YouTube_SiteCloud-AERIAL-Drone-Photography-and-V_Media_IVspDj02u8U_001_1080p.avi"
output_path = "/Users/ajaykasu/Documents/EarthCam/inference_demo.mp4"

def compress_video():
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    
    # Get original properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Target: 640 width (maintain aspect ratio)
    new_width = 640
    new_height = int(height * (new_width / width))
    
    print(f"Compressing to: {new_width}x{new_height}...")
    
    # Video Writer
    # mp4v or avc1 for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize
        resized = cv2.resize(frame, (new_width, new_height))
        out.write(resized)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames}...")
            
    cap.release()
    out.release()
    print("Compression complete.")
    
    # Check size
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"New size: {size_mb:.2f} MB")

if __name__ == "__main__":
    compress_video()
