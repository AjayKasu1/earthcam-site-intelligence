import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

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

st.write("Upload an image or video to detect safety violations and productivity metrics.")

# Sidebar for configuration
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Load Model
@st.cache_resource
def load_model():
    # Look for best.pt in current directory, fallback to yolov8n.pt if not found
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
        
        # Read and display original
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        # Run Inference
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Running detection..."):
                # Convert PIL to CV2
                img_array = np.array(image)
                results = model.predict(img_array, conf=conf_threshold)
                
                # Plot results
                res_plotted = results[0].plot()
                
                # Display Result
                with col2:
                    st.subheader("Detected Objects")
                    st.image(res_plotted, use_container_width=True)
                    
                # Optional: Show counts
                st.info(f"Detections found: {len(results[0].boxes)}")

    # === VIDEO PROCESSING ===
    elif file_type == 'mp4':
        st.subheader("Video Analysis")
        
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30 # fallback
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.text(f"Video Info: {fps} FPS, {total_frames} frames total.")
        
        if st.button("Start Analysis (First 10s)", type="primary"):
            st_frame = st.empty()
            progress_bar = st.progress(0)
            
            # Limit to 10 seconds
            max_frames = fps * 10
            frame_count = 0
            
            while vf.isOpened() and frame_count < max_frames:
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Run inference
                results = model.predict(frame, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                # Convert BGR to RGB for Streamlit
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Update display
                st_frame.image(res_rgb, caption=f"Frame {frame_count}", use_container_width=True)
                
                frame_count += 1
                progress_bar.progress(min(frame_count / max_frames, 1.0))
            
            vf.release()
            st.success("Analysis complete (limit reached).")
