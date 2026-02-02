# Project Overview

In major disasters (earthquakes, floods), cellular networks often collapse, leaving victims with zero internet access. Communication is only possible via low-bandwidth protocols like Bluetooth Low Energy (BLE) or LoRa, which have strict maximum transmission unit (MTU) limits.

**Emergency-LLM solves this by providing:**

* Edge AI Capability: Utilizing Unsloth and 4-bit quantization, the model is optimized to run offline on local devices without needing a cloud connection.

* Byte-Size Optimization: It compresses verbose human speech into ultra-compact packets. Since Bluetooth/LoRa can only send a few hundred bytes at a time, this compression is the difference between a message being "Sent" or "Failed."

* Multi-Protocol Ready: The output is structured to fit perfectly into BLE advertisements or LoRaWAN frames.

# Key Features

*SOTA Fine-Tuning:* Leveraging Gemma-3 (via Unsloth) with 4-bit Quantization (QLoRA) for high-performance inference on consumer-grade hardware.

*Intelligent Compression:* Reduces character count by up to 70% while maintaining semantic integrity.

*Advanced NLP Pipeline:* Custom Turkish/English preprocessing, including emoji removal, abbreviation mapping (e.g., "mah" -> "mahalle"), and tokenization.

*System Simulation:* A mock environment simulating real-world data acquisition:

Automatic GPS Coordinate injection.

Battery status monitoring.

Dynamic media link (Photo/Voice) handling.

Interactive Visualizations: Temporal trend analysis of unique word counts and word frequency distribution.

*Multilingual Support* : The system intelligently handles multiple languages (Turkish/English), making it adaptable for international disaster relief operations.


# Tech Stack 

*Model:* Google Gemma-3 (Fine-tuned with Unsloth)

*Optimization:* PEFT (Parameter-Efficient Fine-Tuning), LoRA, BitsAndBytes.

*Libraries:* transformers, datasets, nltk, pandas, seaborn, wordcloud.

*Environment:* Designed for Google Colab/Kaggle with CUDA support.

# Methodology
*1. Data Engineering*
Preprocessing: Cleaning tweets and emergency logs using RegEx, NLTK Stopwords, and custom Turkish dictionaries.

Feature Extraction: Calculating unique word counts and word frequencies to understand the data distribution before model training.

*2. Model Training (PEFT/LoRA)*
The model was fine-tuned using the following configuration:

Rank (r): 16

Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

Quantization: 4-bit for minimal VRAM usage.

*3. Simulation & Validation*
The project includes a generate_enhanced_message function that simulates a mobile app environment, merging user input with auto-detected device data before sending it to the LLM for optimization.
Dynamic Language Selection: Users can select their preferred language, and the system automatically adjusts the SYSTEM_PROMPT and processing logic to ensure the highest compression efficiency for that specific language.

# Results
The system successfully transforms verbose inputs into "SMS-ready" or "LoRaWAN-friendly" short bursts:

**Original:** "Hi, I'm stuck here in sector 7, near the old bridge, the water is rising. I have two kids with me, Mark has a fever. We need blankets. Coordinates 34.56, -123.45. Battery 5%."

**Optimized:** SOS Sector 7: Flood. 2 Kids (Mark w/ fever). 34.56,-123.45. Need: Rescue/Blankets. Low Bat.


# Project Structure 

├── gemma-3n-E4B-model/     # Saved model & tokenizer
├── data/                   # Dataset files (earthquake logs)
├── notebooks/              # Main logic and visualizations
└── README.md               # Project documentation

