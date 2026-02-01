import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import time

# Set page config
st.set_page_config(
    page_title="EarthCam Site Intelligence",
    page_icon="🏗️",
    layout="wide"
)

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
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    else:
        st.sidebar.warning("Note: Custom 'best.pt' not found. Using standard YOLOv8n for demo.")
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
            # Attempt to use H.264 (avc1) for browser support, fallback to mp4v if needed
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
            
            # Seek to start
            vf.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)
            
            max_frames = fps * 10
            count = 0
            
            while vf.isOpened() and count < max_frames:
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Inference
                results = model.predict(frame, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                # Write to video file
                out.write(res_plotted)
                
                # Update UI (convert BGR to RGB)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(res_rgb, caption=f"Processing: {start_time + count/fps:.1f}s", use_container_width=True)
                
                count += 1
                progress_bar.progress(min(count / max_frames, 1.0))
            
            vf.release()
            out.release()
            
            # Show Result
            st.success("Analysis Complete!")
            st.subheader("Result Playback")
            
            # We need to re-open the file to read bytes for st.video/download
            # Note: OpenCV's 'avc1' sometimes has issues with browsers if ffmpeg isn't perfectly linked.
            # But let's try displaying it.
            try:
                st.video(output_temp.name)
            except:
                st.warning("Could not natively play video, but you can download it below.")
                
            with open(output_temp.name, "rb") as file:
                btn = st.download_button(
                    label="⬇️ Download Analyzed Video",
                    data=file,
                    file_name="earthcam_analysis.mp4",
                    mime="video/mp4"
                )
