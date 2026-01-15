Thermal Rescue Vision 🚁🌡️

Real-time human detection system for autonomous rescue robots using FLIR thermal imagery.

📖 Overview

This project is the computer vision module for an autonomous rescue robot designed to locate survivors in low-visibility environments (smoke, night, debris). Developed as a Final Year Project (PFA) at ENSATB (École Nationale des Sciences Appliquées), it leverages Deep Learning algorithms to process raw thermal data and identify human heat signatures in real-time.

Key Features

Processing: Handling of raw FLIR ADAS v2 thermal datasets.

Architecture: ResNet50 backbone modified for single-channel thermal input (Grayscale).

Optimization: Custom hybrid Data Pipeline (RAM vs VRAM) optimized for consumer hardware (RTX 3060/2050).

Filtered Detection: Logic focusing specifically on "Living Things" (Humans/Animals) while ignoring non-relevant thermal noise.

👥 The Team (ENSATB PFA 2026)

This project is a collaborative effort by engineering students at ENSATB, combining expertise in software, hardware, and robotics:

Youssef Majdhoub – Computer Vision & AI

Oussama Amar – Robotics Integration & Hardware

Amin Saadaoui – Embedded Systems & Control

Academic Supervision:

Pr. Mme Emna Laaridhi – Department of Computer Engineering, ENSATB

⚙️ System Architecture & Optimization Strategy

To process over 10,000 high-resolution thermal images ($640 \times 512$) efficiently on constrained hardware, we engineered a custom two-stage data pipeline. This architecture ensures stability and maximizes throughput across different hardware configurations.

1. Adaptive Memory Management (data_set_manager)

We implemented a Hybrid RAM/VRAM Storage system that dynamically adapts to the available hardware resources:

Low-Resource Configuration (to_GPU=False): Designed for laptops (e.g., RTX 2050 with 4GB VRAM). Data is stored in System RAM (CPU) and batches are streamed to the GPU during training. This strategy mitigates Out-Of-Memory (OOM) exceptions.

High-Performance Configuration (to_GPU=True): Designed for workstations (e.g., RTX 3060 with 12GB+ VRAM). The entire dataset is pre-loaded into VRAM, eliminating CPU-to-GPU transfer latency for maximum training throughput.

Data Quantization: Images are stored as uint8 (0-255) rather than float32, reducing memory footprint by approximately 75% (~10GB savings). Normalization is performed "Just-In-Time" during the forward pass.

2. Deterministic Data Retrieval (data_set_server)

A wrapper around PyTorch's DataLoader that enforces rigorous concurrency constraints:

Constraint: Enforces num_workers=0 for in-memory datasets.

Technical Rationale: Standard multi-processing forks the RAM/CUDA context. When data is already pre-loaded in memory, this forking process leads to memory duplication and potential crashes. This class guarantees thread safety and stability.

📂 Project Structure

PFA2026/
├── archive/                     # Raw Dataset (FLIR ADAS v2)
│   └── FLIR_ADAS_v2/
│       └── images_thermal_train/
│           ├── coco.json        # Original Annotations
│           └── data/            # Thermal Images
├── data_set/
│   ├── human_and_..._data_set.csv # Processed Labels (Generated)
│   └── data_set_creation_script/
│       └── data_integrity_check.py
├── training_scripts/
│   ├── data_handling.py         # 🚀 MAIN PIPELINE (Manager + Server)
│   └── train.py                 # Training Loop
├── setup.py                     # Environment Setup
├── requirements.txt             # Dependencies
└── README.md                    # Documentation



🚀 Quick Start

1. Environment Setup

We utilize a setup script to auto-detect the operating system and configure the GPU environment.

git clone [https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git](https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git)
cd Thermal-Rescue-Vision
python setup.py


2. Generate Dataset Labels

Parse the COCO JSON metadata to generate the training CSV:

# Run the parsing script located in data_set_creation_script/
# This generates: ./data_set/human_and_living_creatures_count_data_set.csv


3. Initialize Data Pipeline

Verify hardware settings and loader functionality before training:

from training_scripts.data_handling import data_set_manager, data_set_server

# A. Initialize Data Manager (Select Mode based on Hardware)
# Use to_GPU=False for 4GB VRAM Laptops
# Use to_GPU=True for 12GB+ VRAM PCs
dataset = data_set_manager(
    csv_file="./data_set/human_and_living_creatures_count_data_set.csv",
    to_GPU=False 
)

# B. Start Delivery Server
loader = data_set_server(dataset, batch_size=24)

# C. Verify Batch Dimensions
images, labels = next(iter(loader))
print(f"Batch Shape: {images.shape}") 
# Output: torch.Size([24, 1, 512, 640])


🛠️ Tech Stack

Core: Python 3.12, PyTorch 2.6

Vision: OpenCV, PIL, Matplotlib

Hardware Acceleration: CUDA 12.4 (Optimized for NVIDIA Ampere Architecture)

Model: ResNet50 (Customized for Single-Channel Thermal Input)

📜 License

Project developed for academic purposes at ENSATB.