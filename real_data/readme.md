## 🔍 Label Verification Methodology

To ensure the integrity of the segmentation masks during the conversion to COCO format, we verified the class mapping indices ($0-4$) through **Data Triangulation**. We cross-referenced three independent sources to confirm the label identities without relying on unverified assumptions.

### The Verified Class Mapping
| ID | Class Name        | Source of Truth |
|:--:|:------------------|:--------------- |
| **0** | Background        | Standard Convention |
| **1** | Fire Extinguisher | Paper (Table I) + Utility Code (Red Channel) |
| **2** | Backpack          | Paper (Table I) + Instance Counts |
| **3** | Hand Drill        | Paper (Table I) + Instance Counts |
| **4** | Survivor          | Paper (Table I) + Thermal Intensity |

### Evidence Sources

#### 1. Academic Source (The Paper)
We referenced **Table I** ("PST900 Dataset Class Imbalance") in the original ICRA 2020 paper. The table lists the foreground classes in a specific column order which matches the pixel indices ($1 \rightarrow 4$).
* **Source:** [PST900: RGB-Thermal Calibration, Dataset and Segmentation Network (Shivakumar et al., 2020)](https://arxiv.org/abs/1909.10980)

#### 2. Statistical Verification (Instance Counting)
We utilized **Dataset Ninja**, an independent dataset auditing tool, to view the total instance counts for each class label. We matched these exact counts to the columns in the official paper to mathematically link the Label ID to the Class Name.
* **Observation:** The class with ID `1` had 1,024 instances, matching the "Fire Extinguisher" count in the paper. The class with ID `2` had 1,676 instances, matching "Backpack".
* **Source:** [Dataset Ninja - PST900 Analysis](https://datasetninja.com/pst900-rgbt)

#### 3. Code & Visual Inspection
We examined the `utilities.py` script from the original authors' toolkit. The visualization function maps Class ID `1` to RGB `(0,0,255)` (Red in BGR) and Class ID `4` to `(255,255,255)` (White).
* **Correlation:** Fire extinguishers are visually red. In thermal imaging, the "Survivor" (heated dummy) is the brightest (white) object. This visual confirmation aligns perfectly with the academic and statistical evidence.