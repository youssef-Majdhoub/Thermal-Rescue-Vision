Thermal Rescue Vision: Autonomous Navigation in Low-Visibility Environments 🚁🌡️

Deep learning-based computer vision system for thermal object detection in rescue robotics.

📖 Operational Context

This project serves as the visual processing unit for autonomous rescue robots operating in confined, low-visibility environments (e.g., collapsed mines, smoke-filled corridors, subway wreckage).

In such scenarios, standard RGB sensors are rendered ineffective by poor lighting, and LiDAR performance degrades due to particulate matter. Radiometric Thermal Imagery (FLIR) is utilized as the primary sensing modality to locate survivors based on heat signatures rather than visual contrast.

📊 Datasets & Training Data

To ensure robust detection in diverse environments, this project utilizes a tiered data strategy, moving from general recognition to specific deployment scenarios.

1. General Recognition: FLIR Thermal Starter Dataset (ADAS)

Role: Primary Baseline.

Objective: Foundational human detection (walking/standing).

Resolution: 640×512 (Matches sensor specifications).

Volume: 10,000+ labeled thermal frames.

2. Pose Robustness: TSF (Thermal Simulated Fall)

Role: Asymmetric Recall & Posture Generalization.

Objective: To distinguish between a "standing rescuer" and a "prone survivor."

Focus: Introduces "Distress" postures (lying prone, crawling, falling) which are absent in standard ADAS datasets. This is critical for the "Living Entities" branch of the network.

3. Deployment Scenario: PST900 RGBT Dataset

Role: Operational Environment Fine-Tuning.

Objective: Domain adaptation for the final deployment environment.

Relevance: Unlike ADAS (outdoors), PST900 focuses specifically on subterranean and confined environments (tunnels, mines, caves).

Processing: Semantic segmentation masks are converted into binary "Existence" targets for our regressor.

Source: PST900 Repository.

🛑 Optimization Strategy

Rescue robotics requires a tailored cost function to address specific failure modes. We implement a hierarchical error minimization strategy:

1. Human Count: Logarithmic Error Minimization (MSLE)

Objective: Minimize relative error. This ensures high precision for small groups (critical for prioritization) while tolerating variance in high-density crowds.

2. Living Entities: Asymmetric Recall Prioritization

Objective: Zero False Negatives (Type II errors).

Methodology: A weighted penalty (10x) is applied specifically to missing a target.

Outcome: The system minimizes the risk of overlooking a survivor (Critical Failure).

🧪 Automated Hyperparameter Search

To scientifically determine the optimal configuration for deployment, the project now includes an Automated Experimentation Engine.

Instead of manual trial-and-error, the Experimentation class performs a structured Grid Search across:

Batch Sizes: Balancing VRAM usage vs. Gradient stability.

Optimizers: Comparing convergence of Adam vs. SGD vs. RMSProp.

Learning Rates: Finding the "Goldilocks zone" for Zero-Gamma convergence.

State Persistence & Crash Recovery

The engine features a Transactional State Manager (experiment_state.pt).

Immutability: The experiment plan (Optimizers/LRs) is "sealed" at start-up. Code changes during a run are ignored to ensure scientific consistency.

Crash Recovery: If the training rig (RTX 3060) loses power, the script automatically detects the state file and resumes exactly where it left off, skipping completed permutations.

⚙️ System Architecture

Experimental Phase 1: Modified ResNet50

We are currently utilizing ResNet50 as our initial baseline architecture to establish performance benchmarks. Future iterations will evaluate lighter, embedded-optimized backbones (e.g., MobileNet, EfficientNet) and Transformers to optimize the Accuracy-Latency trade-off on edge hardware.

The current backbone is structurally adapted for single-channel thermal input with a Zero-Gamma Initialization strategy to prevent gradient explosion during the initial "cold start."

Input: Conv2d(1, 64) for raw thermal tensors (Grayscale).

Output: A Dual-Head Regression Layer yielding 2 Scalar Values:

Human Count: The estimated number of standard standing/walking pedestrians (Rescuers/Bystanders).

Living Things Count: The estimated total number of living entities, specifically capturing prone, crawling, or obscured survivors that standard detection might miss.

Initialization: Residual branches initialized to zero ($\gamma=0$), forcing the network to act as an Identity Map initially.

📂 Project Structure

PFA2026/
├── archive/                     # Raw Dataset Storage
├── data_set/                    # Processed Datasets
├── data_set_creation_script/    # Data Prep Tools
│   ├── data_evaluation_manager.py
│   ├── data_integrity_check.py
│   └── data_set_manager.py
├── evaluation_set/              # Validation Splits
├── falling humans/              # TSF / Pose Data
├── real_data/                   # Deployment Data (PST900)
│   ├── PST900_RGBT_Dataset/
│   ├── PST900_RGBT_Dataset.zip
│   └── readme.md
├── training_scripts/            # Core Logic
│   ├── data_handling.py         # Data Pipeline (Loader & Server)
│   ├── expeimentation.py        # Automated Grid Search Engine
│   └── training_script.py       # ResNet50 Class & Logic
├── .gitignore
├── readme.md
├── requirements.txt
└── setup.py


🚀 Usage Guide

1. Environment Setup

git clone [https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git](https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git)
cd Thermal-Rescue-Vision
pip install -r requirements.txt




2. Running Experiments (Grid Search)

To launch the automated research engine:

from training_scripts.expeimentation import Experimentation
import torch
import os

# Define the Search Space
experiment = Experimentation(
    model_path=os.getcwd(),
    learning_rates=[0.1,0.01,0.001, 0.0001],
    optimizers=[torch.optim.AdamW,torch.optim.Adam, torch.optim.SGD],
    batch_sizes=[8, 16],
    epoch_per_experiment=20,
    device="cuda"
)

# Launch (Auto-resumes if interrupted)
experiment.run_experiments(mode="RTX3060")




3. Inference / Deployment

To load the best-performing model for edge deployment:

from training_scripts.training_script import resnet50_adapted
import os

# Mode 1 = Deployment (Loads from deployment/ folder)
model = resnet50_adapted(home_path=os.getcwd(), mode=1)

# Ready for inference on Jetson/RTX 2050
model.model.eval() 




🛠️ Technology Stack

Core Framework: Python 3.12, PyTorch 2.6

Sensing Modality: FLIR Radiometric Thermal

Acceleration: CUDA 12.4 (Ampere Architecture)

Target Hardware: NVIDIA Jetson Orin / RTX Laptop

👥 Project Team (ENSATB PFA 2026)

Developed by engineering students at the National School of Advanced Sciences and Technologies of Borj Cedria (ENSTAB):

Youssef Majdhoub – Computer Vision & AI Architecture

Oussama Amar – Robotics Integration & Hardware

Amin Saadaoui – Embedded Systems & Control

Academic Supervision:
Pr. Mme Emna Laaridhi – Department of Computer Engineering, ENSTAB

License: Developed for academic purposes at ENSTAB.