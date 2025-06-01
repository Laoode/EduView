<h1 align="center"><a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&weight=500&size=29&duration=1&pause=1000&color=000000&background=fcf06a&vCenter=true&repeat=false&width=941&lines=EduView+Smart+Online+Proctoring+Assist+for+Hybrid+Cheating+Detection" alt="Typing SVG" /></a></h1>

<div align="center">
  <img src="https://github.com/Laoode/EduView/blob/main/images/cyberpunk-object.gif" alt="Banner">
</div>

# Table of Contents

1. [Overview](#overview)
2. [System Overview](#system-overview)
3. [System Architecture](#system-architecture)
4. [User Interface](#user-interface)
5. [Technology Stack](#technology-stack)
6. [Core Components](#core-components)
7. [Summary](#summary)
8. [Installation and Setup](#installation-and-setup)

## Overview
EduView is a smart online proctoring assistant designed to detect potential cheating behaviors during online & offline (hybrid) examinations. The system leverages computer vision, deep learning models, and eye tracking techniques to analyze video feeds from webcams or uploaded media, identifying suspicious activities to assist proctors.

## System Overview
EduView provides a comprehensive solution for online exam proctoring with the following capabilities:
- Real-time processing of webcam feeds
- Analysis of uploaded images and videos
- Three specialized detection models:
    - Classroom behavior detection
    - Cheating behavior detection
    - Eye gaze tracking and analysis
- User-configurable detection thresholds
- Detailed visualization of detection results
The system is implemented as a web application built with [Reflex](https://reflex.dev/), providing an intuitive user interface for proctors to monitor student behavior during online examinations.

## System Architecture
### High-Level Architecture Diagram
<figure className="text-center my-4">
  <img 
    src="https://www.mermaidchart.com/raw/360dddf1-f951-4a5d-bd87-7d8341a0fe15?theme=base&version=v0.1&format=svg" 
    alt="System Architecture"
    className="mx-auto" 
  />
</figure>

The `CameraState` class serves as the central component of the system, coordinating between input sources, detection models, and the user interface. The three detection models provide specialized analysis capabilities, while the UI components offer a comprehensive interface for controlling the system and viewing detection results.

Sources: 

[![Code](https://img.shields.io/badge/object__cheating.py-Lines%201--11-blue?logo=github)](https://github.com/Laoode/EduView/blob/290e2510/object_cheating/object_cheating.py#L1-L11)
<br>
[![Code](https://img.shields.io/badge/object__cheating.py-Lines%2013--53-blue?logo=github)](https://github.com/Laoode/EduView/blob/290e2510/object_cheating/object_cheating.py#L13-L53)

### Detection Workflow
<figure className="text-center my-4">
  <img 
    src="https://www.mermaidchart.com/raw/36f93a58-54c0-4897-917b-91cc0c247634?theme=base&version=v0.1&format=svg" 
    alt="Detection Workflow"
    className="mx-auto" 
  />
</figure
  
The detection workflow starts with the input frame being processed by the CameraState. Based on the selected model, different detection processes are applied. Models 1 and 2 use YOLO-based object detection, while Model 3 leverages the eye tracking subsystem (OpenCV + CNN). The detection results are then processed and displayed in the UI.

Sources: 

[![Code](https://img.shields.io/badge/object__cheating.py-Lines%2013--53-blue?logo=github)](https://github.com/Laoode/EduView/blob/290e2510/object_cheating/object_cheating.py#L13-L53)

## User Interface
The EduView user interface is divided into two main sections that provide comprehensive monitoring and control capabilities:

<div align="center">
  <img src="https://github.com/Laoode/EduView/blob/main/images/app_view.png" alt="UI">
</div>

The left section contains the camera feed, controls for operating the system, and a table displaying detection results. The right section houses panels for adjusting thresholds, viewing statistics, analyzing behaviors, tracking coordinates, and managing inputs.

## Technology Stack
EduView is built using the following technologies:

| 🔧 Component | 💻 Technology |
|-----------|------------|
| 🎨 Frontend Framework | Reflex 0.7.1 |
| 👁️ Computer Vision | OpenCV 4.11.0.86 |
| 🧠 Deep Learning | TensorFlow 2.18.0, Ultralytics 8.3.91 (YOLO) |
| 😊 Face Detection | MediaPipe |
| 📊 Data Processing | NumPy |

Sources: 

[![Requirements](https://img.shields.io/badge/requirements.txt-Dependencies-green?logo=github)](https://github.com/Laoode/EduView/blob/main/requirements.txt)

## Core Components
### CameraState
The `CameraState` class is the central component of the EduView system. It manages:
- Camera feed processing
- Video and image analysis
- Model selection and application
- Detection result storage
  
For more information, see: 

[![Code](https://img.shields.io/badge/camera_state.py-blue?logo=github)](https://github.com/Laoode/EduView/blob/290e2510/object_cheating/states/camera_state.py)

### ThresholdState
The `ThresholdState` class manages detection thresholds, allowing users to adjust:
- Confidence thresholds for detection models
- IoU (Intersection over Union) thresholds for Models 1 and 2
- Duration thresholds for Model 3 (eye tracking)
  
For more information, see:

[![Code](https://img.shields.io/badge/threshold_state.py-blue?logo=github)](https://github.com/Laoode/EduView/blob/290e2510/object_cheating/states/threshold_state.py)

### EyeTracker
The `EyeTracker` component is responsible for:
- Face detection using MediaPipe
- Eye region extraction
- Eye closed detection
- Gaze direction determination
- Coordinate tracking
- Alert generation for suspicious eye movements
  
For more information, see: 

[![Code](https://img.shields.io/badge/eye_tracker.py-blue?logo=github)](https://github.com/Laoode/EduView/blob/290e2510/object_cheating/utils/eye_tracker.py)

## Summary
EduView provides a comprehensive solution for online exam proctoring through its integration of computer vision, deep learning, and eye tracking techniques. The system's modular architecture allows for easy switching between different detection models while maintaining a consistent user experience.

## Installation and Setup
### System Requirements
Before installing EduView, ensure your system meets the following requirements:
#### Hardware Requirements
- Modern CPU (multi-core recommended for real-time analysis)
- At least 8GB RAM (16GB recommended for smooth operation)
- GPU with CUDA support (recommended for faster model inference)
- Webcam for live proctoring (optional if only analyzing uploaded videos)
#### Software Requirements
- Python 3.10 or newer
- Git (for cloning the repository)
- pip (Python package manager)
- Compatible operating system: Windows 10/11, macOS, or Linux

### Installation Process
<figure className="text-center my-4">
  <img 
    src="https://www.mermaidchart.com/raw/c1bb3ef8-4e85-4d41-93ec-535116ce8939?theme=base&version=v0.1&format=svg" 
    alt="Installation Process"
    className="mx-auto" 
  />
</figure

Sources: 

[![Requirements](https://img.shields.io/badge/requirements.txt-Dependencies-green?logo=github)](https://github.com/Laoode/EduView/blob/main/requirements.txt)

#### Step 1: Clone the Repository
Clone the EduView repository from GitHub:
```bash
git clone https://github.com/Laoode/EduView.git
cd EduView
```
#### Step 2: Create a Virtual Environment
It's recommended to use a virtual environment for Python projects to avoid dependency conflicts:
```bash
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```
#### Step 3: Install Dependencies
Install all required packages using pip:
```bash
pip install -r requirements.txt
```
This will install the following key dependencies:
- reflex (v0.7.1): Web framework for the UI
- opencv-python (v4.11.0.86): Computer vision library
- tensorflow (v2.18.0): Machine learning framework
- mediapipe: Face and pose detection
- numpy: Numerical computing
- ultralytics (v8.3.91): For YOLO models

Sources: 

[![Requirements](https://img.shields.io/badge/requirements.txt-Dependencies-green?logo=github)](https://github.com/Laoode/EduView/blob/main/requirements.txt)
#### Step 4: Running the Application
To start the EduView application, run the following command from the project root directory:
```bash
reflex run
```
This will start the development server, and you can access the application by opening a web browser and navigating to `http://localhost:3000` (or the address shown in the terminal).

### Initial Configuration
<figure className="text-center my-4">
  <img 
    src="https://www.mermaidchart.com/raw/55e4a560-2943-4966-a34f-19882acec0a9?theme=base&version=v0.1&format=svg" 
    alt="Initial Configuration"
    className="mx-auto" 
  />
</figure

Sources: 

[![Requirements](https://img.shields.io/badge/requirements.txt-Dependencies-green?logo=github)](https://github.com/Laoode/EduView/blob/main/requirements.txt)

[![.gitignore](https://img.shields.io/badge/.gitignore-File-red?logo=git)](https://github.com/Laoode/EduView/blob/main/.gitignore)

#### Setting Detection Thresholds
After starting the application, you may want to configure detection thresholds to adjust the sensitivity of the detection models. These can be adjusted from the Threshold Panel in the UI:
1. Confidence Threshold: Minimum confidence score for detection (higher values are more strict)
2. IoU Threshold: Intersection over Union threshold for Models 1 and 2
3. Duration Threshold: Time threshold for Model 3 (eye tracking)

#### Selecting a Detection Model
EduView supports three detection models:
1. Model 1: Classroom Behavior Detection - General classroom monitoring
2. Model 2: Cheating Detection - Specific focus on identifying cheating behaviors
3. Model 3: Eye Tracking - Monitors eye movements for suspicious patterns
   
Select the appropriate model from the Controls Panel based on your proctoring needs.

### Video Demo
https://github.com/user-attachments/assets/b495fa4c-ea29-4932-aec8-5279749e8bca

> [!TIP]
> The video above demonstrates a trial run the EduView app. Please note that the playback speed has been increased by **6.6×** and the quality has been reduced to comply with GitHub’s upload size limitations.
> For a clearer and full-resolution version, you can watch it on my **LinkedIn** profile. Alternatively, you may reduce the playback speed on GitHub to **0.25× or 0.5×** for a more natural viewing experience.

### Troubleshooting
#### Common Issues
| **Issue**                      | **Solution**                                                                 |
|-------------------------------|------------------------------------------------------------------------------|
| Missing models                | Run the application once to download models automatically                   |
| Camera not detected           | Check camera permissions and connections                                    |
| Dependencies installation errors | Ensure you're using Python 3.8+ and try installing dependencies one by one |
| `"ModuleNotFoundError"`       | Verify virtual environment is activated and all requirements are installed  |
| Detection directory missing   | The application will create it on first use, or create it manually          |

#### Log Files
Error logs are stored in the application's default logging location. Check these logs for detailed error information if you encounter issues.

### Next Steps
After successful installation and setup, you can proceed to:
- Configure detection thresholds for optimal performance
- Test with different input sources (webcam, images, videos)
- Begin monitoring for suspicious behaviors

> [!NOTE]  
> For more details about core components, detection models, UI components, and advanced configuration — I’m still writing the full documentation on my portfolio website.  
> 🟢 Work in progress: [yudhyprayitno.vercel.app](https://yudhyprayitno.vercel.app/)

---

## 🤝 Contributing

Suggestions, improvements, and contributions are welcome.  
Please open an Issue or submit a Pull Request via GitHub.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📞 Contact

Feel free to reach out if you have any questions or feedback:

[![Instagram](https://img.shields.io/badge/Instagram-%40yudhyprayitno-E4405F?logo=instagram&logoColor=white&style=flat)](https://www.instagram.com/yudhyprayitno)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yudhy%20Prayitno-0077B5?logo=linkedin&logoColor=white&style=flat)](https://www.linkedin.com/in/yudhy-prayitno/)

