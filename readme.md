Thermal Rescue Vision 🚁🌡️

Real-time human detection system for autonomous rescue robots using FLIR thermal imagery.

📖 Overview

This project is the computer vision module for an autonomous rescue robot designed to locate survivors in low-visibility environments (smoke, night, debris). Developed as a Final Year Project (PFA) at ENSATB (École Nationale des Sciences Appliquées), it leverages Deep Learning (YOLOv8) to process raw thermal data and identify human heat signatures in real-time.

Key Features:

Processing of raw FLIR ADAS thermal datasets.

Custom YOLO training pipeline optimized for NVIDIA Ampere GPUs.

Cross-platform deployment environment with auto-detection for CPU/GPU.

Filtered detection logic focusing specifically on "Living Things" (Humans/Animals) while ignoring non-relevant thermal noise.

🛠️ Tech Stack

Core: Python 3.12, PyTorch 2.5

Vision: OpenCV, YOLOv8, Matplotlib

Hardware Acceleration: CUDA 12.4 (Optimized for RTX 3060/2050)

Data Management: Custom scripts for FLIR metadata extraction and normalization.

🚀 Quick Start

We have automated the environment setup to ensure reproducibility across different hardware configurations (from developer laptops to embedded NVIDIA Jetson modules).

1. Clone the Repository

git clone [https://github.com/YourUsername/Thermal-Rescue-Vision.git](https://github.com/YourUsername/Thermal-Rescue-Vision.git)
cd Thermal-Rescue-Vision


2. Auto-Setup (Recommended)

This script will detect your OS, install Python dependencies, and configure GPU support automatically.

Windows:

python setup_env.py


(Follow the on-screen instructions to activate the environment: venv\Scripts\activate)

3. Verification

Run the system check to confirm GPU detection:

venv\Scripts\python main.py


📊 Dataset & Methodology

We utilize the FLIR Thermal Starter Dataset, converting 8-bit thermal JPEGs into YOLO-compatible annotated formats.

Input: FLIR Radiometric JPEGs.

Preprocessing: Grayscale normalization and thermal palette mapping.

Model: YOLOv8 (Nano/Small) selected for high-FPS edge deployment on rescue rovers.

👥 The Team (ENSATB PFA 2026)

This project is a collaborative effort by engineering students at ENSATB, combining expertise in software, hardware, and robotics:

Youssef Majdhoub – Computer Vision & AI

Oussama Amar – Robotics Integration & Hardware

Amin Saadaoui – Embedded Systems & Control

📜 License

Project developed for academic purposes at ENSATB.