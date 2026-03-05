import streamlit as st
import os
import sys
import traceback

print("DIAGNOSTIC: Script start")

def main():
    try:
        st.title("🏗️ EarthCam Site Intelligence Diagnostic")
        st.write("If you can see this, the core Streamlit engine is running.")
        
        print("DIAGNOSTIC: Importing CV2")
        import cv2
        st.success(f"OpenCV loaded: {cv2.__version__}")
        
        print("DIAGNOSTIC: Importing dependencies")
        import numpy as np
        from PIL import Image
        st.success("NumPy and Pillow loaded")
        
        print("DIAGNOSTIC: Importing YOLO")
        from ultralytics import YOLO
        st.success("YOLO Engine loaded")
        
        print("DIAGNOSTIC: Checking for weights")
        if os.path.exists("best.pt"):
            st.info("Found best.pt weights")
            model = YOLO("best.pt")
            st.success("Model loaded from best.pt")
        else:
            st.warning("best.pt not found, using yolov8n.pt")
            model = YOLO("yolov8n.pt")
            st.success("Model loaded from yolov8n.pt")
            
        st.balloons()
        st.write("### All systems go! Ready for inference.")
        
    except Exception as e:
        print(f"DIAGNOSTIC ERROR: {str(e)}")
        print(traceback.format_exc())
        st.error(f"Startup Error: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
