#  Emergency-LLM: Edge-AI for Disaster Communication

**Emergency-LLM** is a high-performance, offline-capable language model designed to ensure survival communication when cellular networks collapse. Developed using real-world data from the **2023 Turkey Earthquake (Hatay)**, it optimizes critical messages into ultra-compact packets for transmission over low-bandwidth protocols like **LoRa** and **BLE (Bluetooth Low Energy)**.

---

##  The Problem: The "Zero-Internet" Crisis
During major disasters (earthquakes, floods), infrastructure often fails. Victims are left with zero internet. Low-bandwidth communication (LoRa/BLE) becomes the only lifeline, but these protocols have strict **MTU (Maximum Transmission Unit)** limits. A verbose message will fail; a compressed one will save lives.

##  Key Features
*   **Edge AI Optimization:** Powered by **Gemma-3** and fine-tuned via **Unsloth (4-bit QLoRA)**, allowing high-speed inference on local devices without cloud dependency.
*   **Intelligent Semantic Compression:** Reduces character count by up to **70%** while preserving 100% of the critical information (Location, Health, Needs).
*   **Battle-Tested Dataset:** Trained on high-stress, non-standard grammar logs from the 2023 Turkey Earthquake, making it resilient to typos and local dialects.
*   **Multilingual Support:** Intelligent processing for both **Turkish and English** relief operations.
*   **System Simulation:** Real-time injection of GPS coordinates, battery status, and media links (Photo/Voice) into the communication pipeline.

---

##  Tech Stack
*   **Base Model:** Google Gemma-3 (8B/4B/27B variants)
*   **Fine-Tuning:** Unsloth, PEFT (Parameter-Efficient Fine-Tuning), LoRA.
*   **Quantization:** BitsAndBytes (4-bit) for minimal VRAM usage.
*   **NLP Tools:** NLTK, Transformers, RegEx, Custom Turkish NLP dictionaries.
*   **Visualization:** Seaborn, WordCloud, Pandas for temporal trend analysis.

---

##  Methodology

### 1. Data Engineering
We process chaotic emergency logs by:
*   Removing noise (excessive emojis, filler words).
*   Mapping abbreviations (e.g., "mah" ➔ "mahalle").
*   Analyzing word frequency distribution to identify survival-critical tokens.

### 2. Model Training (PEFT/LoRA)
The model was fine-tuned with a focus on **efficiency** and **precision**:
*   **Rank (r):** 16
*   **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
*   **Quantization:** 4-bit for deployment on consumer-grade smartphones/laptops.

### 3. Simulation Environment
The `generate_enhanced_message` function mimics a mobile app environment, merging user input with auto-detected metadata:
`Input + GPS + Battery Level + Language Selection ➔ LLM ➔ Compact SOS Packet`

---

##  Ethics & Privacy

* All data derived from the 2023 Earthquake has been handled with extreme care, focusing strictly on extracting life-saving information while respecting the gravity of the source material.

* Developed for: Disaster Relief, Search & Rescue Operations, and Edge-AI Research.

* Hardware Requirement: Compatible with 8GB VRAM (Kaggle/Colab/Local GPU).

---

##  Performance: Compression in Action

| Input Type | Original Message | Optimized (LoRa-Ready) |
| :--- | :--- | :--- |
| **Verbose User Input** | "Hi, I'm stuck here in sector 7, near the old bridge, the water is rising. I have two kids with me, Mark has a fever. We need blankets. Coordinates 34.56, -123.45. Battery 5%." | **SOS Sector 7: Flood. 2 Kids (Mark w/ fever). 34.56,-123.45. Need: Rescue/Blankets. Low Bat.** |

---

##  Project Structure
```text
├── gemma-3-E4B-model/      # Fine-tuned weights & tokenizer
├── data/                   # Earthquake logs & cleaned datasets
├── notebooks/              # Training logic, EDA & visualizations
└── README.md               # Documentation
