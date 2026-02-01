# 🏗️ EarthCam Site Intelligence & Safety Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

![Project Demo](demo_preview.gif)

An end-to-end Computer Vision & Data Analytics pipeline designed to monitor construction site safety, productivity, and logistics automatically. Built as a portfolio project for a **Junior Data Scientist** role.

[🎥 Watch Full Demo Video (33MB)](colab_outputs/demo_full.mp4)

## 🎯 Project Goal
Transform raw video footage into actionable business insights by detecting 26 granular classes and mapping them to 3 key "Business Pillars":

1.  **Safety**: Hardhats, Safety Vests, Masks, Person categorization.
2.  **Productivity**: Excavators, Dump Trucks, Concrete Mixers.
3.  **Security/Logistics**: Vehicles, Machinery Access.

## 🚀 Features
-   **Custom AI Model**: Fine-tuned **YOLOv8-Nano** on a custom dataset (717 images, 50 epochs).
-   **Business Logic Layer**: Python dictionary mapping for high-level KPI aggregation.
-   **Interactive Web App**: Streamlit dashboard for real-time inference on images and videos.
-   **Data Pipeline**: Automated export of detection logs to **SQLite** database and **CSV** for Power BI integration.

## 📂 Project Structure
```bash
├── app.py                      # Streamlit interactive web application
├── best.pt                     # Custom trained YOLOv8 model weights
├── earthcam_project_colab.py   # Original Colab inference script (Batch Processing)
├── colab_outputs/              # Results from Google Colab training run
│   ├── inference_demo.mp4      # Compressed video demonstration
│   ├── final_site_report.csv   # Analytics data for Power BI
│   └── earthcam_analytics.db   # SQL Database with logs
├── requirements.txt            # Python dependencies for Cloud deployment
└── README.md                   # Project documentation
```

## 🛠️ Installation & Usage

### 1. Run Locally
Clone the repository and install dependencies:
```bash
git clone https://github.com/AjayKasu1/earthcam-site-intelligence.git
cd earthcam-site-intelligence
pip install -r requirements.txt
```

Run the Streamlit App:
```bash
streamlit run app.py
```

### 2. Google Colab Workflow
Check `EarthCam_Project_colab.ipynb` for the full training pipeline (Mounts Google Drive, trains model, exports SQL).

## 📊 Methodology
1.  **Data Cleaning**: Mapped 26 granular classes (e.g., 'dump truck', 'excavator') into Executive Categories.
2.  **Training**: Used Transfer Learning on YOLOv8n (Nano) for edge-deployment efficiency.
3.  **Inference**: Processed video frames to extract timestamped detection counts.
4.  **Analytics**: Structured unstructured video data into a Queryable SQL Database.

## 👨‍💻 Author
**Ajay Kasu**
*Junior Data Scientist Candidate*
