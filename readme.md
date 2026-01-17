Thermal Rescue Vision: Autonomous Navigation in Low-Visibility Environments 🚁🌡️

Deep learning-based computer vision system for thermal object detection in rescue robotics.

📖 Operational Context

This project serves as the visual processing unit for autonomous rescue robots operating in confined, low-visibility environments (e.g., collapsed mines, smoke-filled corridors, subway wreckage). In such scenarios, standard RGB sensors are rendered ineffective by poor lighting, and LiDAR performance degrades due to particulate matter. Radiometric Thermal Imagery (FLIR) is utilized as the primary sensing modality.

🛑 Optimization Strategy

Unlike general-purpose object detection, rescue robotics requires a tailored cost function to address specific failure modes. We implement a hierarchical error minimization strategy:

1. Human Count: Logarithmic Error Minimization

Objective: Minimize relative error rather than absolute error to maintain high sensitivity in low-density scenarios.

Methodology: We utilize MSLE (Mean Squared Logarithmic Error). This metric penalizes prediction errors based on ratio. This ensures the model maintains high precision when detecting small groups (critical for rescue prioritization) while tolerating acceptable statistical variance in high-density crowds.

2. Living Entities: Asymmetric Recall Prioritization

Objective: Eliminate False Negatives (Type II errors) in binary existence classification.

Methodology: An Asymmetric "Existence" Loss. The cost function applies a significant penalty weight (100x) specifically to False Negatives (predicting zero occupancy when a target is present). This bias ensures the system favors recall over precision for the binary "Safe/Unsafe" classification.

Outcome: The system minimizes the risk of overlooking survivors (Critical Failure) while maintaining a manageable false-positive rate.

⚙️ System Architecture: Modified ResNet50

The core architecture is a ResNet50 backbone, structurally adapted to process single-channel Grayscale inputs (Thermal intensity). The network operates as a Multi-Task Regressor using a Zero-Gamma Initialization strategy.

1. Structural Modifications

Input Layer Adaptation: Conv2d(3, 64) $\rightarrow$ Conv2d(1, 64).

Purpose: Direct ingestion of raw thermal intensity tensors without artificial RGB upsampling.

Dual-Head Output: Linear(2048, 1000) $\rightarrow$ Linear(2048, 2).

Head A: Human Count Regression (Log-Space).

Head B: Living Entity Existence (Binary-Weighted).

2. Initialization Protocol (Zero-Gamma)

Standard weight initialization introduces random noise, which can obscure faint thermal signatures during early training epochs. We employ a deterministic Signal-Preservation Strategy:

Input Layer (Conv1) Weights $\approx$ 1.0:

Mechanism: Unbiased Intensity Accumulation.

Purpose: Forces the initial layer to act as a pass-through filter, ensuring the entire thermal energy budget reaches the backbone without attenuation.

Residual Branch Suppression (Weights $\approx$ 0.0):

Mechanism: We initialize the scale parameters ($\gamma$) of the final Batch Normalization layer in each residual block to zero.

Theory: This effectively sets the residual function $F(x) = 0$, reducing the block to an Identity Mapping ($y = x$).

Purpose: This creates a "Skeptical Baseline," forcing the model output to [0, 0] initially. The network only deviates from this baseline when the gradient signal (driven by the asymmetric loss) is sufficiently strong, thereby reducing random false positives during the initial convergence phase.

💾 State Management & Reproducibility

The project implements a hardware-agnostic serialization system designed to ensure reproducibility and seamless transition between training (RTX 3060) and deployment (RTX 2050) environments.

1. Checkpoint Separation

We distinguish between training states and deployment artifacts:

Training Checkpoints (training/): Serializes Model Weights, Optimizer State, Epoch index, and Loss metrics. This enables precise resumption of the optimization process after interruptions.

Deployment Artifacts (deployment/): Serializes only the optimized model weights (state_dict). This minimizes storage footprint for the target embedded hardware.

2. Device-Agnostic Loading

The loading mechanism implements a "CPU Layover" strategy (map_location='cpu').

Problem: PyTorch checkpoints store device affinity (e.g., cuda:0), causing runtime errors when loaded on devices with different GPU configurations.

Solution: The class automatically remaps tensors to the CPU during deserialization before moving them to the local accelerator, ensuring portability across heterogeneous hardware.

📉 Mathematical Formulation

Loss Functions

The operational requirements are enforced via a custom composite loss function.

Human Count (MSLE):

$$L_{human} = \text{mean}((\log(1+y) - \log(1+\hat{y}))^2) \times 100$$

Living Entity (Asymmetric Weighted Loss):

$$\text{Coeff} = \begin{cases}
100.0 & \text{if } Target \> 0 \land Prediction \approx 0 \text{ (Critical Miss)} \\
1.0 & \text{otherwise}
\end{cases}$$

$$L_{safety} = \text{mean}(\text{Coeff} \times (\text{LogDiff})^2)$$

Adaptive Memory Pipeline

To support high-resolution thermal imagery ($640 \times 512$) across varying hardware tiers:

High-Throughput Mode (RTX 3060): Pre-loads batches (Size 8) to VRAM, utilizing Gradient Accumulation to simulate larger effective batch sizes.

Memory-Constrained Mode (RTX 2050): Utilizes reduced batch sizes (Size 2) with Just-In-Time (JIT) RAM streaming to prevent Out-Of-Memory (OOM) errors.

📂 Project Structure

PFA2026/
├── archive/                     # Dataset Storage
├── data_set_creation_script/    # Data Prep Tools
│   ├── data_integrity_check.py
│   └── data_set_manager.py
├── training_scripts/            # Core Logic
│   ├── data_handling.py         # Data Pipeline (Loader & Server)
│   └── training_script.py       # Main Entry Point (Model & Loop)
├── .gitignore
├── readme.md
└── requirements.txt


🚀 Usage Guide

1. Environment Setup

git clone [https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git](https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git)
cd Thermal-Rescue-Vision
pip install -r requirements.txt



2. Training

The system creates hardware-specific execution profiles. To initiate training:

from resnet50_adapted import resnet50_adapted
import os

# Initialize the model (Mode 0 = Training)
# Directory structure for checkpoints is automatically generated
model = resnet50_adapted(home_path=os.getcwd(), mode=0)

# Execute training loop
# Profiles: "RTX3060" (Batch 8), "RTX2050" (Batch 2), "CPU" (Batch 4)
model.train(epochs=10, mode="RTX3060")



3. Inference / Deployment

To load the optimized weights for deployment:

# Mode 1 = Deployment
# Automatically scans 'deployment/model_data/' for the optimal weight file
model = resnet50_adapted(home_path=os.getcwd(), mode=1)

# The model is initialized and ready for inference
# model.model.eval() 



🛠️ Technology Stack

Core Framework: Python 3.12, PyTorch 2.6

Sensing Modality: FLIR Radiometric Thermal

Acceleration: CUDA 12.4 (Ampere Architecture)

Target Hardware: NVIDIA Jetson / RTX Laptop

👥 Project Team (ENSATB PFA 2026)

This project is developed by engineering students at the National School of Advanced Sciences and Technologies of Borj Cedria (ENSATB):

Youssef Majdhoub – Computer Vision & AI Architecture

Oussama Amar – Robotics Integration & Hardware

Amin Saadaoui – Embedded Systems & Control

Academic Supervision:
Pr. Mme Emna Laaridhi – Department of Computer Engineering, ENSATB

License: Developed for academic purposes at ENSATB.