import streamlit as st
import os
import sys

print("APP_DEBUG: Script execution started")

# Set page config
st.set_page_config(
    page_title="EarthCam Site Intelligence",
    page_icon="🏗️",
    layout="wide"
)

print("APP_DEBUG: Attempting to import OpenCV")
try:
    import cv2
    print(f"APP_DEBUG: OpenCV imported successfully, version: {cv2.__version__}")
except Exception as e:
    print(f"APP_DEBUG: OpenCV import FAILED: {str(e)}")
    st.error(f"Critical Dependency Error: {e}")
    st.stop()

print("APP_DEBUG: Importing other libraries")
import tempfile
import numpy as np
from PIL import Image
import time

print("APP_DEBUG: UI rendering started")
st.title("🏗️ EarthCam Site Intelligence Demo")
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

st.write("Upload an image or video. Videos will be trimmed to 10 seconds for analysis.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Show Demo GIF if available
if os.path.exists("demo_preview.gif"):
    st.sidebar.image("demo_preview.gif", caption="Real-time Demo")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Load Model
@st.cache_resource
def load_model():
    print("APP_DEBUG: Inside load_model - importing YOLO")
    from ultralytics import YOLO
    print("APP_DEBUG: YOLO imported")
    if os.path.exists("best.pt"):
        print("APP_DEBUG: Loading best.pt")
        return YOLO("best.pt")
    else:
        print("APP_DEBUG: best.pt not found, loading yolov8n.pt")
        return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose a file...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # === IMAGE PROCESSING ===
    if file_type in ['jpg', 'jpeg', 'png']:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Running detection..."):
                img_array = np.array(image)
                results = model.predict(img_array, conf=conf_threshold)
                res_plotted = results[0].plot()
                with col2:
                    st.subheader("Detected Objects")
                    st.image(res_plotted, use_container_width=True)
                st.info(f"Detections found: {len(results[0].boxes)}")

    # === VIDEO PROCESSING ===
    elif file_type == 'mp4':
        st.subheader("Video Analysis")
        
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        vf = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        st.info(f"Video Duration: {duration:.2f} seconds ({total_frames} frames @ {fps} FPS)")
        
        # Timeline Slider for Trimming
        start_time = 0
        if duration > 10:
            st.warning("Video exceeds 10 seconds. Please select the 10s segment you want to analyze.")
            start_time = st.slider("Start Time (seconds)", 0, int(duration) - 10, 0)
        
        if st.button(f"Analyze 10s Clip (Start: {start_time}s)", type="primary"):
            st_frame = st.empty()
            progress_bar = st.progress(0)
            
            # Output setup
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            # Attempt to use browser-compatible codec
            # 'avc1' is H.264 (best for web), but requires openh264 on the server
            # 'mp4v' is a good fallback that most browsers can download and play locally
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
            
            # Seek to start
            vf.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)
            
            max_frames = fps * 10
            count = 0
            
            status_text = st.empty()
            while vf.isOpened() and count < max_frames:
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Inference
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Write to video file
                out.write(res_plotted)
                
                # Update UI (convert BGR to RGB)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(res_rgb, caption=f"Analyzing: {start_time + count/fps:.1f}s", use_container_width=True)
                
                count += 1
                progress_bar.progress(min(count / max_frames, 1.0))
            
            vf.release()
            out.release()
            
            # Show Result
            st.success("Analysis Complete!")
            
            # Provide Download instantly
            with open(output_temp.name, "rb") as file:
                video_bytes = file.read()
                st.download_button(
                    label="⬇️ Download Analyzed Video",
                    data=video_bytes,
                    file_name="earthcam_analysis.mp4",
                    mime="video/mp4",
                    type="primary"
                )
            
            # Try to show playback (might still fail in some browsers due to codec, but download will work)
            st.subheader("Result Preview")
            st.video(video_bytes)
