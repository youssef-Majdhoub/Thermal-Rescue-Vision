Thermal Rescue Vision: Tunnel & Debris Operations 🚁🌡️

Autonomous rescue vision system designed for zero-visibility environments.

📖 Mission Profile: The "Dark Tunnel" Scenario

This project acts as the sensory cortex for a rescue robot operating in Tunnel-Like Environments (collapsed mines, smoke-filled corridors, subway wreckage). In these conditions, standard RGB cameras are useless due to lack of light, and LiDAR struggles with smoke particulate. Thermal Imagery (FLIR) is the only viable modality.

🛑 The Operational Doctrine

Unlike standard object detection, a Rescue Robot has two conflicting objectives. We have engineered our model to respect a hierarchical "Cost of Error":

1. Human Count → Doctrine: "Relative Precision"

The Logic: "Missing 1 person in a crowd of 100 is acceptable; missing 1 person in a group of 4 is a failure."

The Solution: We utilize MSLE (Mean Squared Logarithmic Error). This penalizes errors based on ratio rather than absolute difference. It forces the model to be hyper-accurate with small groups (readiness) while tolerating statistical noise in large crowds.

2. Living Things → Doctrine: "The Zero-to-One Criticality"

The Logic: "Predicting 0 when there is 1 is a fatal error (Abandonment). Predicting 1 when there is 0 is a logistical error (Search Fatigue)."

The Solution: An Asymmetric "Existence" Loss. The model faces a massive penalty (100x) for predicting "Zero" on an inhabited frame (The "Dangerous Gambit"), but is trained with standard precision elsewhere to prevent "Search Fatigue" (False Positives on empty frames).

Outcome: The robot is terrified of missing a single life form but disciplined enough not to brake for every hot rock.

⚙️ System Architecture: The "Bold" Adaptation

We utilize a ResNet50 backbone, surgically modified to accept single-channel Grayscale inputs (Thermal intensity). The network is trained as a Multi-Task Regressor using a novel "Zero-Backbone" strategy.

1. The Model Surgery

Input (Thermal Eye): Conv2d(3, 64) $\rightarrow$ Conv2d(1, 64). Adapts the network to ingest raw thermal intensity tensors directly.

Output (Dual Head): Linear(2048, 1000) $\rightarrow$ Linear(2048, 2).

Head A (Human Count): Optimized for Logarithmic Precision.

Head B (Living Existence): Optimized for Zero-False-Negative Recall.

2. Identity-Driven Initialization ("The Zero-Backbone")

Standard weight initialization introduces random noise. In thermal rescue scenarios, this noise can "filter out" faint heat signatures before the model learns to recognize them. We employ a Signal-Wire Initialization strategy to prevent this data fading.

Input Layer (Conv1): Weights = 1.0

Theory: Unbiased Intensity Accumulation.

Mechanism: We initialize the first layer as a pure intensity accumulator.

Critical Logic: Random weights act as a filter. By using 1.0, we force the layer to pass the entire thermal energy budget into the backbone. This ensures that faint "Living Thing" signals are not inadvertently discarded by a bad random initialization before the network converges.

Backbone Layers: Weights = 0.0

Theory: The Silent Highway.

Mechanism: By zeroing the convolution weights ($F(x) \to 0$), the residual blocks act as Identity Mappings ($y \to x$).

Critical Logic: This collapses the deep network into a linear path initially. It allows the raw signal from the Input Layer to reach the classifiers without distortion, solving the "Vanishing Gradient" problem for critical, low-contrast thermal targets.

Output Layer (Head): Weights = 0.0

Theory: Null Hypothesis vs. Paranoid Loss.

Mechanism: The model defaults to predicting [0, 0] (Empty Road).

Critical Logic: Our Asymmetric Loss is extremely generous (it rewards finding things). If we initialized randomly, the model might hallucinate obstacles to satisfy this loss. By forcing the initial state to "Zero," we create a Skeptical Baseline. The model only predicts "Danger" when the gradient signal (driven by the 100x penalty on actual misses) is strong enough to overcome this zero-inertia. This counter-balances the False Positives inherent in safety-critical loss functions.

📉 Technical Implementation: The Loss Math

The "Operational Doctrine" is enforced mathematically via a custom composite loss function.

Human Count Math (MSLE)

$$L_{human} = \text{mean}((\log(1+y) - \log(1+\hat{y}))^2) \times 100$$

Why: Logarithms compress large errors but magnify small errors relative to the target. This aligns perfectly with the "Relative Precision" doctrine.

Living Creature Math (Asymmetric Safety)

$$\text{Coeff} = \begin{cases} 
100.0 & \text{if } Target > 0 \land Prediction \approx 0 \text{ (Critical Miss)} \\
1.0 & \text{otherwise}
\end{cases}$$

$$L_{safety} = \text{mean}(\text{Coeff} \times (\text{LogDiff})^2)$$

Why: The dynamic coefficient acts as a "Safety Gate," forcing the optimizer to prioritize Recall over Precision specifically for the "Zero-to-One" transition.

💾 Adaptive Memory Management

To handle high-resolution thermal images ($640 \times 512$) required to spot small limbs in large tunnels, we engineered a custom memory pipeline in data_handling.py:

Hybrid RAM/VRAM Storage:

Laptop Mode (RTX 2050): Stores data in System RAM, streams to GPU just-in-time. Prevents OOM on 4GB cards.

PC Mode (RTX 3060): Pre-loads data to VRAM for maximum throughput.

Data Quantization: Images stored as uint8 to maximize RAM density (4x compression), expanded to float32 Just-In-Time.

Safety Wrapper: Custom DataLoader wrapper (data_set_server) enforces thread safety to prevent memory forking crashes on Windows/Linux boundaries.

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


🚀 Quick Start

1. Environment Setup

git clone [https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git](https://github.com/youssef-Majdhoub/Thermal-Rescue-Vision.git)
cd Thermal-Rescue-Vision
python setup.py


2. Verify Data Pipeline

Check if your hardware (Laptop vs PC) is detected correctly. Note the import path change due to project structure:

from training_scripts.data_handling import data_set_manager

# Initialize (False for Laptop, True for PC)
dataset = data_set_manager(
    csv_file="./archive/FLIR_ADAS_v2/labels_thermal_train.csv",
    to_GPU=False 
)
print(f"Loaded {len(dataset)} thermal frames.")


3. Start Training

from training_scripts.training_script import training_manager

# Start the training loop
# Modes: "RTX3060", "RTX2050", "CPU"
training_manager.train(mode="RTX3060", epochs=10)


🛠️ Tech Stack

Core: Python 3.12, PyTorch 2.6

Sensor: FLIR Radiometric Thermal

Hardware Acceleration: CUDA 12.4 (Ampere Architecture)

Deployment: Optimized for NVIDIA Jetson / RTX Laptop Roaming

👥 The Team (ENSATB PFA 2026)

This project is a collaborative effort by engineering students at ENSATB, combining expertise in software, hardware, and robotics:

Youssef Majdhoub – Computer Vision & AI

Oussama Amar – Robotics Integration & Hardware

Amin Saadaoui – Embedded Systems & Control

Academic Supervision:
Pr. Mme Emna Laaridhi – Department of Computer Engineering, ENSATB

📜 License: Project developed for academic purposes at ENSATB.