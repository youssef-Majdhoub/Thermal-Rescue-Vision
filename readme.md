# Thermal Rescue Vision: Embedded Perception Module for Autonomous Search and Rescue (SAR) Robotics 🚁🌡️

An advanced embedded computer vision framework developed to power autonomous terrestrial rescue robots operating under degraded visibility conditions (smoke, total darkness, dust, and concrete debris). 

This project serves as the core perception engine for tactical engineering solutions in hazardous fields, systematically isolating, identifying, and tracking human thermal signatures to assist emergency responders and Civil Protection teams.

---

## 📖 Operational Context & Technical Challenges

During emergency operations (e.g., structural collapses, active fires, industrial accidents), standard RGB cameras fail due to light dependency, while LiDAR systems suffer from reflection distortions caused by high particulate suspension (dust, smoke). 

This architecture leverages **Infrared Thermography** to sense natural body heat emissions, bypassing visual barriers. The core engineering challenges addressed are:
* **Thermal Clutter Filtering:** Distinguishing true human signatures from high-density environmental heat anomalies (e.g., thermal echoes, warm machinery, solar-heated asphalt, fire pits).
* **Edge Computing Optimization:** Fine-tuning object detection networks to optimize the trade-off between geometric precision, high sensitivity, and rapid inference times on resource-constrained embedded robotic hardware.

---

## 📊 Dataset Ingestion & Curation Strategy

To build domain generalization and prevent model bias across varied search environments, the training pipeline handles hybrid data ingestion from multiple reference sets (including **PST900**, **Teledyne FLIR ADAS**, and the **Khan Falling Humans** video dataset).

### Data Distribution Layout
The unified database is strictly segregated into three distinct subsets to guarantee an impartial performance evaluation and eliminate overfitting:
* **Training Set (80%–85%):** Used by the network architectures to extract specific human thermal features and iteratively update internal weights.
* **Validation Set (~10%):** Utilized during the training phase to tune hyperparameters and implement early stopping mechanics.
* **Test Set (5%–10%):** Composed exclusively of unencountered, novel images to act as an objective evaluation benchmark under real-world conditions.

### Training Hyperparameters & Data Augmentation
To maximize performance within physical hardware constraints (trained using an active **RTX GPU** execution state), the configuration parameters were structured as follows:
* **Epochs:** 200 (`epochs=200`) to guarantee deep structural convergence of complex thermal silhouettes.
* **Batch Size:** 8 (`batch=8`) optimized for hardware Video RAM limits.
* **Input Resolution:** Resized to 640x640 pixels (`imgsz=640`) to maintain structural fidelity while keeping processing footprints low.
* **Data Pipelines:** Accelerated via CUDA execution layers (`device="cuda"`), supported by 4 concurrent loader processes (`workers=4`) and active RAM caching (`cache=True`).
* **Real-Time Data Augmentations:** Implemented to improve model robustness against background clutter, including spatial rotations (`degrees=5`), scaling transformations (`scale=0.9`), and advanced **Copy-Paste** synthesis (`copy_paste=0.5`) to overlay human thermal silhouettes onto novel unconditioned backbones.

---

## 🧪 Model Performance & Validation Benchmarks

Model tracking focuses on three key metrics: **Precision** (minimizing false alarms from ambient heat sources), **Recall** (ensuring no human victims are left behind), and **mAP** (overall localization and classification accuracy).

### Validation Run Insights
* **Overall Metrics:** The initial evaluation yielded an F1-Score of 0.72, with an overall tracking metric of **mAP@0.5 = 0.708**. 
* **Geometric Precision:** Tight bounding box framing registered an **mAP@0.5:0.95 = 0.42**, showing room for spatial refinement around low-contrast silhouettes.
* **Confusion Matrix Breakdown:** The core validation pass successfully logged **305 True Positives (TP)**. It registered 60 False Positives (FP), and identified 155 False Negatives (FN) where individuals blended heavily into thermal background noise or presented atypical postures (prone/collapsed states).

---

## ⚖️ Architectural Trade-off: YOLOv8s vs. YOLO26m

An extensive architectural comparison was conducted to evaluate the optimal model for embedded deployment. The test results are detailed in the index ledger below:

| Performance Metric | YOLOv8s | YOLO26m | Operational Winner |
| :--- | :---: | :---: | :---: |
| **Max Precision** | 1.00 | 1.00 | Tie |
| **Max Recall** | **0.82** | 0.78 | **YOLOv8s** |
| **mAP@0.5** | **0.725** | 0.708 | **YOLOv8s (+0.017)** |
| **mAP@0.5:0.95** | ~0.40 | **~0.45** | **YOLO26m** |
| **GPU Inference Latency** | ~1.4 ms | **~0.9 ms** | **YOLO26m (-36%)** |
| **Real-Time Embedded Fit** | Partially | **Fully** | **YOLO26m** |

### 🛠️ Strategic Engineering Selection
While **YOLO26m** demonstrated clear advantages in raw compute speed (reducing inference latency by 36% down to 0.9 ms) and a slight lead in bounding box geometry, **YOLOv8s was chosen as the primary deployment architecture**. 

In high-stakes search-and-rescue configurations, **Recall** is the absolute highest priority metric. Missing a survivor (a False Negative) carries catastrophic real-world consequences. YOLOv8s delivers a superior max recall score of **0.82** compared to YOLO26m's 0.78, making it the more reliable choice for preserving human lives.

---

## 📂 Project Repository Tree

```text
Thermal-Rescue-Vision/
├── data_set/                   # Unified and standardized tensor splits
├── data_set_creation_script/   # Curation, parsing, and format transformation tools
│   ├── data_evaluation_manager.py
│   └── data_integrity_check.py
├── training_scripts/           # Computational logic
│   ├── data_handling.py        # Stream loading, tensor formatting, and data feeding
│   ├── expeimentation.py       # Automated grid-search and checkpoint recovery engine
│   └── training_script.py      # Architectural definitions and training mechanics
├── requirements.txt            # System dependencies
└── setup.py                    # Distribution configurations
```

## 🚀 Quickstart Guide

### 1. Dependencies and Environment Setup

```bash
git clone https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git
cd Thermal-Rescue-Vision
pip install -r requirements.txt
```
---

## 🛑 Project Appraisals & Long-Term Horizons

### ⚡ Core Technical Advantages
* **Visibility-Independent Tracking:** Delivers zero-light operational capabilities, tracking targets seamlessly through total darkness and active dense smoke.
* **Low-Latency Edge Inference:** Optimized execution layer achieving highly responsive `1.4 ms` inference profiles for real-time stream processing.
* **Modular Integration Architectures:** Native compatibility with standardized mobile terrestrial robotic platform base configurations.

### ⚠️ Current Limitations & Future Work

> **Thermal Interference Challenge**
> High-temperature ambient environments (e.g., burning debris, active structural fires) emit intense infrared background noise that can degrade victim contrast bounds.

* **Sensor Fusion Pipeline:** Upcoming integration streams will combine the infrared vision network with **3D LiDAR sensor frameworks** to filter out environmental noise via geometric depth filtering.
* **Transition to Local Edge AI:** Migrating compute workloads directly onto the robot's onboard processing hardware (e.g., **NVIDIA Jetson** platforms). This eliminates telemetry communication dependencies in enclosed cave systems or subterranean networks.

---

## 👥 Project Engineering Team

| Team Member | Operational Role & Focus |
| :--- | :--- |
| **Youssef Majdoub** | Computer Vision & Deep Learning Architecture |
| **Amine Saadaoui** | Embedded Systems & Control Logic |
| **Oussama Ammar** | Robotic Integration & Hardware Framework |

### 🎓 Academic Supervision
> **Pr. Emna Aridhi**
> *Département Technologies Avancées et Digitalisation de l'Industrie*
> **École Nationale des Sciences et Technologies Avancées à Borj Cédria (ENSTAB)**

---

<p align="center">
  <i>Developed for academic and research evaluation under the official guidelines of the University of Carthage.</i>
</p>
