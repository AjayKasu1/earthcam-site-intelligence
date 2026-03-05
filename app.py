import streamlit as st
import os
import sys
import tempfile
import numpy as np
import time
from PIL import Image

# Set page config
st.set_page_config(
    page_title="EarthCam Site Intelligence",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡️ EarthCam AI")
    st.image("demo_preview.gif", caption="Safety & Productivity Monitoring")
    st.info("This system uses fine-tuned YOLOv8 to detect site activity and safety violations.")
    st.markdown("---")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.45)
    st.markdown("---")
    st.caption("Developed for Site Intelligence Pipeline v1.0")

st.title("🏗️ EarthCam Site Intelligence Demo")

# Lazy Load YOLO to prevent startup hang
@st.cache_resource
def get_model():
    # Only import heavy dependencies when caching starts
    try:
        import cv2
        from ultralytics import YOLO
        if os.path.exists("best.pt"):
            return YOLO("best.pt")
        else:
            return YOLO("yolov8n.pt")
    except Exception as e:
        return e

# Main UI
tab1, tab2 = st.tabs(["🎥 Video Analysis", "📸 Image Analysis"])

with tab1:
    uploaded_video = st.file_uploader("Upload Site Footage", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video:
        # Load model only on demand
        with st.spinner("Initializing AI Engine... (First time may take a minute)"):
            model_or_err = get_model()
            if isinstance(model_or_err, Exception):
                st.error(f"Failed to load AI Engine: {model_or_err}")
                st.stop()
            else:
                model = model_or_err

        # Temporary save for processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        import cv2 # Lazy import
        vf = cv2.VideoCapture(tfile.name)
        fps = vf.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        st.info(f"Video Duration: {duration:.2f} seconds ({frame_count} frames @ {fps:.0f} FPS)")
        
        if duration > 10:
            st.warning("Video exceeds 10 seconds. Please select the 10s segment you want to analyze.")
            start_time = st.slider("Start Time (seconds)", 0.0, max(0.0, duration - 10.0), 0.0)
        else:
            start_time = 0.0

        if st.button(f"Analyze 10s Clip (Start: {start_time:.1f}s)", type="primary"):
            progress_bar = st.progress(0)
            st_frame = st.empty()
            
            # Output setup
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
            
            # Seek to start
            vf.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)
            
            max_frames = int(fps * 10)
            count = 0
            
            while vf.isOpened() and count < max_frames:
                ret, frame = vf.read()
                if not ret: break
                
                # Inference
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Write to video
                out.write(res_plotted)
                
                # Update UI
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(res_rgb, caption=f"Analyzing: {start_time + count/fps:.1f}s", use_container_width=True)
                
                count += 1
                progress_bar.progress(min(count / max_frames, 1.0))
            
            vf.release()
            out.release()
            
            st.success("Analysis Complete!")
            
            # Instant Download
            with open(output_temp.name, "rb") as file:
                video_bytes = file.read()
                st.download_button(
                    label="⬇️ Download Analyzed Video",
                    data=video_bytes,
                    file_name="earthcam_analysis.mp4",
                    mime="video/mp4",
                    type="primary"
                )
            
            st.subheader("Result Preview")
            st.video(video_bytes)

with tab2:
    uploaded_image = st.file_uploader("Upload Site Photo", type=['jpg', 'jpeg', 'png'])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Detect Activity", type="primary"):
            with st.spinner("Analyzing..."):
                model_or_err = get_model()
                if isinstance(model_or_err, Exception):
                    st.error(f"Error: {model_or_err}")
                else:
                    results = model_or_err.predict(image, conf=conf_threshold)
                    res_plotted = results[0].plot()
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st.image(res_rgb, caption="Result", use_container_width=True)
                    st.success("Detection Complete!")
