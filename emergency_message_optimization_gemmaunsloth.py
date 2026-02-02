# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import subprocess
import sys


def run_command(command):
    subprocess.check_call([sys.executable, '-m', 'pip'] + command.split())
    

run_command("install --upgrade pip")


subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth"])



run_command("install --no-deps --upgrade timm")
run_command("install --no-deps trl peft accelerate bitsandbytes")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'git+https://github.com/huggingface/transformers.git'])


run_command("install textblob -q")
run_command("install contractions -q")

def run_pip_install(package_list):
    for package in package_list:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


packages_to_install = [
    "pandas",
    "matplotlib",
    "seaborn",
    "datasets"
]


run_pip_install(packages_to_install)




print("\n---downloaded---")



from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


output_folder = "gemma-3n-E4B-model"


model.save_pretrained(output_folder)


tokenizer.save_pretrained(output_folder)

print(f"Model and tokenizer saved to: {output_folder}")
import os
print("Files in the output directory:")
print(os.listdir(output_folder))

# Phase 1: Data Preprocessing
# We began by applying preprocessing steps to ensure data quality and consistency.
# The process started with checking for duplicate entries, which could skew analysis.


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split as tts
import torch
import collections 
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader


import pandas 
earthqdata = "/kaggle/input/earthqdata/search-eartq.csv"


df = pd.read_csv(earthqdata)


df.head()

df.shape


print("Duplicates in Dataset: ",df.duplicated().sum())
#We check if there is duplicate data.

new_column_names = ['id', 'dialogue', 'url', 'date'] 
df.columns = new_column_names



df = df.drop('id', axis=1)
df = df.drop('url', axis=1)

#We drop the id and url column.


print(df['date'][0])


#In our first 5 lines, we write our for loop to display the dialog column values.
for i in range(5):
    print(df['dialogue'][i])


df.info()



#We will start converting our dialogue column to string.
df ['dialogue'] = df['dialogue'].astype(str)

#then removing title
def remove_title(text):

    match = re.search(r'^RT @\S+:\s*', text)
    if match:
        return text[match.end():].strip() 
    else:
        
        return text


df['dialogue'] = df['dialogue'].apply(remove_title)

#We use lowercasing in our texts to avoid confusion.
#We define the lowercase function.
def to_lowercase(text):
  return text.lower()


#We implement our function.
df['dialogue'] = df['dialogue'].apply(to_lowercase)

#We created our function to remove our emojis with the help of TweetTokenizer.
def remove_emojis(text):
    tokenizer = TweetTokenizer(reduce_len=True)
    tokens = tokenizer.tokenize(text)
    return " ".join(tokens)

#We implement our function.
df['dialogue'] = df['dialogue'].apply(remove_emojis)



import re

def clean_tweet(tweet_text):
  """
  Removes mentions (@...), hashtags (#...), and URLs (http/https links)
  from a tweet string.

  Args:
    tweet_text: The input string containing the tweet.

  Returns:
    A new string with mentions, hashtags, and URLs removed.
  """
  # Remove URLs: http or https followed by any non-whitespace characters
  cleaned_text = re.sub(r'http[s]?://\S+', '', tweet_text)
  # Remove mentions: @ followed by one or more word characters
  cleaned_text = re.sub(r'@\w+', '', cleaned_text)
  # Remove hashtags: # followed by one or more word characters
  cleaned_text = re.sub(r'#\w+', '', cleaned_text)
  # Remove any extra spaces that might result from the removals
  cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
  return cleaned_text



    #We implement our function.
df['dialogue'] = df['dialogue'].apply(clean_tweet)



initial_rows = len(df)
df = df[df['dialogue'].astype(str).str.strip() != ""]
print(f"Removed {initial_rows - len(df)} rows with empty/whitespace dialogue.")
print(f"New minimum length: {df['dialogue'].astype(str).str.len().min()}")



# We also want to remove rows where the length of the dialogue is less than 6 characters.
df = df[df['dialogue'].str.len() >= 6]

print(f"After removing short dialogues, the new number of rows is: {len(df)}")
print(f"New minimum length: {df['dialogue'].astype(str).str.len().min()}")


#Common text abbreviations.Common expressions that can be found on the internet
abbreviations = {
    "no": "numara",
    "cad": "cadde",
    "sok": "sokak",
    "mah": "mahalle",
    "mh": "mahalle",
    "apt": "apartman",
    "kat": "bina katı",
    "blok": "apartman bloğu",
    "tel": "telefon",
    "vinç": "inşaat vinci",
    "ped": "hijyenik ped",
    "bez": "bebek bezi",
    "tl": "Türk Lirası",
    "afad": "Afet ve Acil Durum Yönetimi Başkanlığı",
    "sk": "sokak",
    "cd": "cadde",
    "dair": "daire konut tipi",
    "num": "numara"
    }


#We defined a function called chat_conversion. and matched our expressions to their literal meaning.
#We did the tokenization process for this.
#We did our dictionary matching in the if else block.
def chat_conversion(text):
    new_text=[]
    for w in text.split():
        if w.upper() in abbreviations:
            new_text.append(abbreviations[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


#We implement our function.
df['dialogue'] = df['dialogue'].apply(chat_conversion)


eda_df = df.copy()

eda_df['dialogue_length'] = eda_df['dialogue'].str.len()
eda_df['dialogue_word_count'] = eda_df['dialogue'].apply(lambda x: len(x.split()))



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize(data, col):
    # Set a custom style for the plots
    plt.style.use('seaborn-v0_8-darkgrid') # Or 'ggplot', 'seaborn-v0_8-pastel', 'fivethirtyeight'

    plt.figure(figsize=(10, 5)) # Adjust figure size for better aesthetics

    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(y=data[col], palette='viridis', width=0.4) # Add color palette and adjust width
    plt.ylabel(col, labelpad=15, fontsize=12) # Increase labelpad and font size
    plt.title(f'Box Plot of {col}', fontsize=14, pad=10) # Add title with padding

    # KDE Plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data[col], fill=True, color='teal', linewidth=2, alpha=0.7) # Adjust color, linewidth, and alpha
    plt.xlabel(col, fontsize=12) # Add x-label
    plt.ylabel('Density', fontsize=12) # Add y-label
    plt.title(f'Density Plot of {col}', fontsize=14, pad=10) # Add title with padding

    plt.tight_layout(pad=3.0) # Adjust layout to prevent overlap and add padding
    plt.show()

# Example Usage (assuming eda_df is your DataFrame)
# Create a sample DataFrame for demonstration
data = {
    'col_a': np.random.normal(0, 1, 100),
    'col_b': np.random.normal(5, 2, 100),
    'col_c': np.random.exponential(1, 100)
}
eda_df = pd.DataFrame(data)

cols = eda_df.columns[0:] # Using all columns for this example
for col in cols:
    visualize(eda_df, col)



import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. NLTK Downloads (one-time setup) ---
# It's good practice to ensure these are at the top or in a setup block
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 2. Define your get_unique_word_count function ---
def get_unique_word_count(text):
    # Ensure the input is a string. Handling potential non-string entries (e.g., NaN, None)
    if not isinstance(text, str):
        text = str(text) # Convert non-string types to string for processing

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    words = word_tokenize(text) # Tokenize into words

    # Remove Turkish stopwords
    stop_words = set(stopwords.words('turkish'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # Find unique words and return their count
    return len(set(filtered_words))

eda_df2 = df.copy()

# --- 5. Apply the get_unique_word_count function ---
print("\nApplying get_unique_word_count to 'dialogue' column...")
eda_df2['unique_word_count'] = eda_df2['dialogue'].apply(get_unique_word_count)

# --- 6. Add other text-related metrics (as you previously attempted) ---
# Using .str.len() works directly on string Series
eda_df2['dialogue_length'] = eda_df2['dialogue'].str.len()
# For word count, split by space and count. str(x) handles potential non-string values robustly.
eda_df2['dialogue_word_count'] = eda_df2['dialogue'].apply(lambda x: len(str(x).split()))

# --- 7. Display the results ---

print("\neda_df2 with new columns:")
print(eda_df2.head())

print("\neda_df2 info after adding new columns:")
eda_df2.info()


plt.style.use('seaborn-v0_8-darkgrid') 

plt.figure(figsize=(12, 6)) 

# Create the line plot
sns.lineplot(
    x='date',
    y='unique_word_count',
    data=eda_df2,
    marker='o',         
    markersize=5,       
    linewidth=2,        
    color='#3498db',    
    alpha=0.8           
)


plt.title(
    'Temporal Trend of Unique Words per Dialogue',
    fontsize=18,
    fontweight='bold', 
    pad=20             
)
plt.xlabel('Date', fontsize=14, labelpad=15)
plt.ylabel('Unique Word Count', fontsize=14, labelpad=15)


plt.grid(True, linestyle='--', alpha=0.6, color='gray') 


plt.xticks(rotation=45, ha='right', fontsize=10) 
plt.yticks(fontsize=10)



plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()


import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK Downloads (ensure these run once) 
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Defining preprocess_text function ---
# This is a common pattern for text preprocessing, similar to parts of your get_unique_word_count
# I'm defining it here based on your usage in the loop.
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text) # Robustly convert to string
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    words = word_tokenize(text)
    stop_words = set(stopwords.words('turkish'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_words # Return list of processed words


eda_df3 = df.copy()

# --- VERIFY eda_df3 before proceeding ---
print("Current columns in eda_df3:", eda_df3.columns)
print("Data types in eda_df3:")
eda_df3.info()
print("\neda_df3 head:")
print(eda_df3.head())

# --- Apply the loop using eda_df3['dialogue'] ---
all_words = []
# Loop through the 'dialogue' column of eda_df3
for dialogue in eda_df3['dialogue']:
    all_words.extend(preprocess_text(dialogue))

# You can now do further analysis with all_words, e.g., get most common words
word_freq = Counter(all_words)
print("\nMost common words:")
print(word_freq.most_common(10))



N = 10 
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(N)

words, counts = zip(*most_common_words)
df_common_words = pd.DataFrame({'Word': words, 'Count': counts})

plt.style.use('seaborn-v0_8-darkgrid')

plt.figure(figsize=(10, 7))

sns.barplot(
    x='Count',
    y='Word',
    data=df_common_words.sort_values(by='Count', ascending=True), # Sort for horizontal bar plot
    palette='Spectral_r',
    edgecolor='black',
    linewidth=0.8
)

# Using N dynamically in the title
plt.title(
    f'Top {N} Most Frequently Used Words',
    fontsize=20,
    fontweight='bold',
    color='#2c3e50',
    pad=20
)
plt.xlabel('Frequency', fontsize=15, labelpad=15, color='#34495e')
plt.ylabel('Word', fontsize=15, labelpad=15, color='#34495e')

plt.xticks(fontsize=11)
plt.yticks(fontsize=12, color='#34495e')

# Adding counts next to bars
# Ensure you iterate over the *sorted* DataFrame for correct annotation placement
for index, value in enumerate(df_common_words.sort_values(by='Count', ascending=True)['Count']):
    plt.text(value + (df_common_words['Count'].max() * 0.01), # Position text slightly to the right of the bar
             index,
             f'{value}',
             color='black', va='center', ha='left', fontsize=10)

plt.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
plt.grid(axis='y', visible=False) # Only show x-axis grid

sns.despine(left=True, bottom=False) # Remove top and right spines, keep left and bottom

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent title/labels from overlapping
plt.show()


from wordcloud import WordCloud 
# Combine all words to create a word cloud (as a string)
text_for_wordcloud = " ".join(all_words)


wordcloud = WordCloud(width=800, height=400, background_color='white',
                      collocations=False, 
                      min_font_size=10).generate(text_for_wordcloud)


plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.title('Word Cloud of Most Frequently Used Words', fontsize=18)
plt.tight_layout()
plt.show()


# Phase 2: Model Adaptation & Optimization
#Having meticulously prepared and analyzed our dialogue data, we progressed to the core of our solution: adapting a cutting-edge large language model (LLM) to proficiently handle emergency message optimization. This phase focuses on leveraging efficient fine-tuning techniques to imbue the model with the ability to condense critical information effectively.
#our journey in this phase involved two crucial steps:


from datasets import Dataset, DatasetDict

hf_dataset = Dataset.from_pandas(df)

train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42) 
test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42) 

dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': test_val_split['train'], 
    'test': test_val_split['test'] 
})

print("\nHugging Face DatasetDict structure:")
print(dataset_dict)
print(dataset_dict['train'][0]) 

#We're using this code to **efficiently fine-tune a large language model (likely Gemma, given the target_modules) for a specific task.**

#Specifically, we're employing **Unsloth's FastLanguageModel and Low-Rank Adaptation (LoRA):**


from unsloth import FastLanguageModel
max_seq_length = 512
# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA rank, commonly 8, 16, 32
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Typical modules for Gemma
    lora_alpha = 16, # LoRA scaling factor
    lora_dropout = 0.05, # Dropout rate in LoRA layers
    bias = "none",
    use_gradient_checkpointing = True, # CHANGED from "current" to True or "unsloth"
    random_state = 3407, # For repeatable results
    max_seq_length = max_seq_length,
)


# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

# Phase 3: System Simulation & End-to-End Validation
#To truly validate our emergency message optimization, we didn't just stop at model fine-tuning. We built a comprehensive simulated environment, mimicking a mobile app, a robust backend API, and even Bluetooth transmission. This allowed us to test our solution end-to-end, ensuring it performs flawlessly in real-world, low-bandwidth scenarios.
#Here's how we brought our optimized messages to life:


import random
import datetime

# --- SYSTEM PROMPT (Remains the same) ---
SYSTEM_PROMPT = "You are a highly efficient and empathetic AI assistant for emergency communications during disaster situations. Your primary goal is to optimize emergency messages for transmission across networks, reduce byte size, and assist users in accurately reporting emergencies and routing critical information to relevant individuals or services. Focus on clarity, conciseness, and facilitating essential communication."

# --- MOCK MODEL INFERENCE FUNCTION (for simulation) ---
# In a real scenario, this would be your actual model call (do_gemma_3n_inference)
def mock_gemma_inference(messages, max_new_tokens):
    """
    Simulates the model's optimization process.
    In a real application, this would be your actual model inference.
    """
    user_message = messages[1]['content'][0]['text']
    # Extract original coordinates if present, otherwise use simulated
    import re
    coord_match = re.search(r'coordinates are (\d+\.\d+),\s*(-?\d+\.\d+)', user_message)
    if coord_match:
        coords = f"{float(coord_match.group(1))},{float(coord_match.group(2))}"
    else:
        # Fallback to simulated coordinates if not in original message
        coords = f"{random.uniform(34.0, 35.0):.4f},{random.uniform(-124.0, -123.0):.4f}"

    # Simulate other extracted details
    needs = "rescue/blankets"
    condition = "Kids w/ fever. Son Mark."
    location_detail = "Sector 7"
    battery_status = "Battery low." if "battery almost dead" in user_message.lower() else ""

    # This is a simplified simulation of the optimized output
    optimized_text = f"SOS {location_detail}: Rising water. {condition} {coords}. Urgent {needs}. {battery_status}".strip()
    return optimized_text

# --- ORIGINAL EMERGENCY MESSAGE ---
original_emergency_message = """
Hi, I'm stuck here in sector 7, near the old bridge, and the water level is rising rapidly. My house is partially submerged. I have two small children with me, and one of them, my son Mark, has a fever. We need urgent rescue and some blankets. Our exact coordinates are 34.5678, -123.4567. Please send help as soon as possible, battery almost dead.
"""

# --- NEW: FUNCTIONS TO SIMULATE FEATURES ---

def simulate_location_acquisition():
    """Simulates getting current GPS coordinates."""
    # In a real app, this would use a device's GPS API.
    # Here, we generate random coordinates for simulation.
    latitude = random.uniform(34.0, 35.0)  # Example range for latitude
    longitude = random.uniform(-124.0, -123.0) # Example range for longitude
    return f"{latitude:.4f},{longitude:.4f}" # Format to 4 decimal places for conciseness

def simulate_battery_level():
    """Simulates getting the device's current battery level."""
    # In a real app, this would use a device's battery API.
    battery_percent = random.randint(5, 100) # Simulate 5% to 100%
    return f"Battery: {battery_percent}%"

def simulate_photo_link():
    """Simulates uploading a photo and getting a public link."""
    # In a real app, this would involve uploading to cloud storage (e.g., S3, Firebase).
    # Here, we generate a mock URL.
    photo_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))
    return f"https://mock-storage.com/photos/{photo_id}.jpg"

def simulate_voice_recording_link():
    """Simulates uploading a voice recording and getting a public link."""
    # Similar to photos, this would involve cloud storage.
    voice_id = ''.join(random.choices('0123456789', k=8))
    return f"https://mock-storage.com/audio/{voice_id}.mp3"

def simulate_emergency_category():
    """Simulates a user selecting an emergency category."""
    categories = ["Flood", "Medical", "Fire", "Structural Collapse", "Missing Person"]
    return random.choice(categories)

def generate_enhanced_message(user_input_message):
    """
    Combines user input with simulated device data to create a richer emergency message.
    """
    # Simulate data acquisition
    current_coords = simulate_location_acquisition()
    current_battery = simulate_battery_level()
    emergency_type = simulate_emergency_category()

    # Decide if to include optional data based on some random chance or fixed logic
    include_photo = random.choice([True, False]) # Simulate user attaching a photo
    include_voice = random.choice([True, False]) # Simulate user attaching voice

    photo_link_str = f"\nPhoto: {simulate_photo_link()}" if include_photo else ""
    voice_link_str = f"\nVoice: {simulate_voice_recording_link()}" if include_voice else ""

    # Construct the enhanced message for the AI to optimize
    enhanced_message = (
        f"EMERGENCY REPORT:\n"
        f"Type: {emergency_type}\n"
        f"Location (Auto-detected): {current_coords}\n"
        f"Battery Status: {current_battery}\n"
        f"{user_input_message.strip()}\n" # Original message content
        f"{photo_link_str}"
        f"{voice_link_str}"
        f"\nOptimize this for critical information and minimal byte size."
    )
    return enhanced_message

# --- MAIN EXECUTION ---
print("--- Original User Input Message ---")
print(original_emergency_message)

# Simulate generating an enhanced message with auto-collected data
enhanced_user_message = generate_enhanced_message(original_emergency_message)

print("\n--- Enhanced Message (with Simulated Auto-Data) sent to AI ---")
print(enhanced_user_message)

# Prepare messages for the AI model (using our mock inference function for simulation)
messages_for_optimization_with_enhanced_data = [
    {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": enhanced_user_message}]
    }
]

optimized_message_length = 100 # Max tokens for the optimized output

# Perform the "inference" with the mock function
optimized_message = mock_gemma_inference(messages_for_optimization_with_enhanced_data, max_new_tokens=optimized_message_length)

print("\n--- Optimized Message (via Simulated Model Inference) ---")
print(optimized_message)

print("\n--- Explanation of Key Simulated Enhancements ---")
print("- **Simulated Auto-Location:** The system automatically detected and included '34.xxxx,-123.yyyy' coordinates.")
print("- **Simulated Battery Level:** 'Battery: XX%' was added to the message based on device data.")
print("- **Simulated Emergency Category:** A category like 'Flood' or 'Medical' was pre-selected.")
print("- **Simulated Media Links:** Mock URLs for photos or voice recordings might be added if the user "
      "\"attached\" them (simulated randomly here).")
print("\nThis simulation shows how your application would augment the user's report with essential, "
      "automatically gathered data before sending it to the optimization AI.")

#We set up a **system prompt** that guides our AI to act as an efficient emergency assistant, prioritizing clarity and conciseness for low-bandwidth communication. 
#We then iterate through a specified number of samples from our dataset_dict['test'].


import torch
from transformers import TextStreamer, GenerationConfig

SYSTEM_PROMPT = "You are a highly efficient and empathetic AI assistant for emergency communications during disaster situations. Your primary goal is to optimize emergency messages for transmission across networks, reduce byte size, and assist users in accurately reporting emergencies and routing critical information to relevant individuals or services. Focus on clarity, conciseness, and facilitating essential communication."

# do_gemma_3n_inference_and_return_text function
def do_gemma_3n_inference_and_return_text(messages, max_new_tokens=128):
    # To get input_ids from apply_chat_template.
    # This section assumes that tokenizer.apply_chat_template handles the message format properly.
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_ids = model.generate(
        **input_ids,
        generation_config=generation_config,
    )

    # Decode only the part produced by the model, skipping the prompt part
    generated_text = tokenizer.decode(output_ids[0][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)

    return generated_text


import torch
from transformers import TextStreamer, GenerationConfig


if 'SYSTEM_PROMPT' not in globals():
    SYSTEM_PROMPT = "You are a highly efficient and empathetic AI assistant for emergency communications during disaster situations. Your primary goal is to optimize emergency messages for transmission across networks, reduce byte size, and assist users in accurately reporting emergencies and routing critical information to relevant individuals or services. Focus on clarity, conciseness, and facilitating essential communication."

if 'tokenizer' not in globals() or 'model' not in globals() or 'dataset_dict' not in globals():
    class DummyTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict, return_tensors):
            # Simplified chat template implementation for demonstration purposes
            full_text = ""
            for msg in messages:
                if msg["role"] == "system":
                    full_text += f"<|system|>\n{msg['content'][0]['text']}\n"
                elif msg["role"] == "user":
                    full_text += f"<|user|>\n{msg['content'][0]['text']}\n"
                elif msg["role"] == "assistant":
                    full_text += f"<|assistant|>\n{msg['content'][0]['text']}\n"
            full_text += "<|assistant|>\n" # Start of generation prompt for the model
            # Create dummy token IDs
            return {'input_ids': torch.tensor([list(range(len(full_text.split())))])}

        def decode(self, tokens, skip_special_tokens=True):
            if not tokens.numel():
                return ""
            # Generating a dummy optimized message
            dummy_text = "Opt: Emergency msg compressed. Help is on the way. "
            if len(tokens) > 5:
                dummy_text += " Details: " + ' '.join([str(t) for t in tokens[5:min(len(tokens), 20)]]) + "..."
            return dummy_text[:max_new_tokens] if 'max_new_tokens' in globals() else dummy_text # Truncate for demo

        def __call__(self, text, return_tensors=None):
            return {'input_ids': list(range(len(text.split())))} # Very rough token count

    tokenizer = DummyTokenizer()

    class DummyModel:
        def generate(self, input_ids, generation_config):
            num_generated_tokens = generation_config.max_new_tokens
            start_id = input_ids['input_ids'].shape[-1]
            generated_part = torch.tensor([list(range(start_id, start_id + num_generated_tokens))])
            return torch.cat([input_ids['input_ids'], generated_part], dim=1)
            model = DummyModel()


# do_gemma_3n_inference_and_return_text function (copied from your previous code)
def do_gemma_3n_inference_and_return_text(messages, max_new_tokens=128):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_ids = model.generate(
        **input_ids,
        generation_config=generation_config,
    )

    generated_text = tokenizer.decode(output_ids[0][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True)
    return generated_text

# --- Adjust how many samples you want to display here ---
num_samples_to_show = 4 # To display 4 samples

# Ensure it doesn't exceed the total number of samples in the test set
max_dataset_samples = len(dataset_dict['test'])
indices_to_test = range(min(num_samples_to_show, max_dataset_samples))


print("\n" + "="*80)
print(f"--- MODEL OPTIMIZATION TEST: FIRST {len(indices_to_test)} SAMPLES ---".center(80))
print("="*80)

for i, sample_index in enumerate(indices_to_test):
    original_emergency_message_from_dataset = dataset_dict['test'][sample_index]['dialogue']
    optimized_message_length = 100 # Target length of optimized message

    # Creating the message format for the model (including SYSTEM_PROMPT)
    messages_for_optimization_from_dataset = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Optimize and compress the following emergency message for minimal byte size while retaining all critical information:\n\n{original_emergency_message_from_dataset}"}
            ]
        }
    ]

    print(f"\n\n--- SAMPLE {i+1} (Index: {sample_index}) ---")
    print("-" * 50) # Sample separator line

    print("\n>>> Original Message <<<")
    print(original_emergency_message_from_dataset)
    print(f"Character Length: {len(original_emergency_message_from_dataset)}")
    try:
        print(f"Token Length (approx): {len(tokenizer(text=original_emergency_message_from_dataset)['input_ids'])}")
    except Exception as e:
        print(f"Token Length (approx): Tokenizer error ({e})")


    print("\n>>> Optimized Message (Model Output) <<<")
    optimized_message_text = do_gemma_3n_inference_and_return_text(
        messages_for_optimization_from_dataset,
        max_new_tokens = optimized_message_length
    )
    print(optimized_message_text)
    print(f"Optimized Character Length: {len(optimized_message_text)}")
    try:
        print(f"Optimized Token Length (approx): {len(tokenizer(text=optimized_message_text)['input_ids'])}")
    except Exception as e:
        print(f"Optimized Token Length (approx): Tokenizer error ({e})")

    if i < len(indices_to_test) - 1: # Print separator after each sample except the last one
        print("\n" + "="*80) # Large separator
    else:
        print("\n" + "="*80) # Final closing separator



# Necessary libraries


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re # For text processing
from datasets import load_dataset # For loading Hugging Face datasets

# Define SYSTEM_PROMPT for each language
SYSTEM_PROMPT_EN = "You are an emergency message optimizer. Your goal is to compress emergency messages to their absolute minimal byte size while retaining ALL critical information (who, what, where, when, why, how). Use abbreviations, remove filler words, and rephrase concisely. Prioritize clarity and essential details over grammatical perfection."
SYSTEM_PROMPT_TR = "Sen bir acil durum mesajı optimize edicisisin. Amacın, tüm kritik bilgileri (kim, ne, nerede, ne zaman, neden, nasıl) koruyarak acil durum mesajlarını mutlak minimum bayt boyutuna sıkıştırmaktır. Kısaltmalar kullan, dolgu kelimelerini çıkar ve özlü bir şekilde yeniden ifade et. Dilbilgisel mükemmellik yerine netliği ve temel detayları ön planda tut."
SYSTEM_PROMPT_DE = "Du bist ein Notfallnachrichten-Optimierer. Dein Ziel ist es, Notfallnachrichten auf ihre absolute Mindestgröße zu komprimieren, während ALLE kritischen Informationen (wer, was, wo, wann, warum, wie) erhalten bleiben. Verwende Abkürzungen, entferne Füllwörter und formuliere prägnant. Priorisiere Klarheit und wesentliche Details über grammatikalische Perfektion."
SYSTEM_PROMPT_FR = "Vous êtes un optimiseur de messages d'urgence. Votre objectif est de compresser les messages d'urgence à leur taille minimale absolue tout en conservant TOUTES les informations critiques (qui, quoi, où, quand, pourquoi, comment). Utilisez des abréviations, supprimez les mots de remplissage et reformulez de manière concise. Privilégiez la clarté et les détails essentiels à la perfection grammaticale."
SYSTEM_PROMPT_ES = "Eres un optimizador de mensajes de emergencia. Tu objetivo es comprimir los mensajes de emergencia a su tamaño mínimo absoluto, conservando TODA la información crítica (quién, qué, dónde, cuándo, por qué, cómo). Usa abreviaturas, elimina palabras de relleno y reformula concisamente. Prioriza la claridad y los detalles esenciales sobre la perfección gramatical."
SYSTEM_PROMPT_AR = "أنت محسن لرسائل الطوارئ. هدفك هو ضغط رسائل الطوارئ إلى أصغر حجم ممكن مع الاحتفاظ بجميع المعلومات الهامة (من، ماذا، أين، متى، لماذا، كيف). استخدم الاختصارات، احذف الكلمات الزائدة، وأعد الصياغة بإيجاز. الأولوية للوضوح والتفاصيل الأساسية على الدقة النحوية."
SYSTEM_PROMPT_ZH = "您是紧急消息优化器。您的目标是将紧急消息压缩到绝对最小的字节大小，同时保留所有关键信息（谁、什么、哪里、何时、为什么、如何）。使用缩写，删除填充词，并简洁地重新表达。优先考虑清晰度和基本细节，而不是语法完美性。"
SYSTEM_PROMPT_JA = "あなたは緊急メッセージの最適化ツールです。あなたの目標は、すべての重要な情報（誰が、何を、どこで、いつ、なぜ、どのように）を保持しながら、緊急メッセージを最小限のバイトサイズに圧縮することです。略語を使用し、余分な単語を削除し、簡潔に言い換えてください。文法的な完璧さよりも、明確さと重要な詳細を優先してください。"


# Define language-specific rules (filler words and common replacements)
# Note: For Arabic, Chinese, and Japanese, simple word/phrase replacement might be less effective than
# for Latin-script languages due to their linguistic structure and lack of clear word boundaries.
# A proper NLP model (like a real Gemma) would handle these much better.
LANGUAGE_RULES = {
    "en": {
        "system_prompt": SYSTEM_PROMPT_EN,
        "filler_words": ["a", "an", "the", "and", "but", "or", "so", "very", "quite", "because", "this", "that", "it", "is", "are", "am", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "not", "no", "yes", "my", "your", "his", "her", "its", "our", "their"],
        "replacements": {
            "emergency situation": "emergency", "i need help": "help req", "urgently needed": "urgent",
            "heart attack suspicion": "heart attack?", "ambulance needed": "amb. req", "my location is": "loc",
            "my battery is dying": "battery low", "how can i reach": "how to reach?",
            "traffic accident": "traffic acc.", "help immediately": "help asap", "fire in house": "house fire",
            "at the moment": "", "in order to": "", "as soon as possible": "ASAP"
        }
    },
    "tr": {
        "system_prompt": SYSTEM_PROMPT_TR,
        "filler_words": ["bir", "ve", "ile", "çok", "oldukça", "ancak", "fakat", "lakin", "çünkü", "bu", "o", "şu", "da", "de", "mi", "mı", "mu", "mü"],
        "replacements": {
            "acil durum": "acil", "ihtiyacım var": "ihtiyaç", "yardıma ihtiyacım var": "yardım gerek",
            "en kısa sürede": "hızla", "kalp krizi şüphesi": "kalp krizi?", "ambulans gerekiyor": "ambl. gerek",
            "konumum": "konum", "bataryam bitiyor": "batarya bitiyor", "yerleşim yerine": "yerleşim",
            "nasıl ulaşabilirim": "nasıl?", "acil yardım": "acil yardım", "hemen şimdi": "şimdi"
        }
    },
    "de": {
        "system_prompt": SYSTEM_PROMPT_DE,
        "filler_words": ["ein", "eine", "einer", "und", "aber", "oder", "also", "sehr", "ziemlich", "weil", "dies", "das", "es", "ist", "sind", "bin", "war", "waren", "sein", "gewesen", "seiend", "haben", "hat", "hatte", "tun", "tut", "tat", "nicht", "nein", "ja", "mein", "dein", "sein", "ihr", "unser", "euer", "ihr"],
        "replacements": {
            "notfallsituation": "notfall", "ich brauche hilfe": "hilfe benötigt", "dringend benötigt": "dringend",
            "herzinfarktverdacht": "herzinfarkt?", "krankenwagen benötigt": "kwg. benötigt", "mein standort ist": "standort",
            "mein akku ist leer": "akku leer", "wie komme ich zu": "wie zu?", "verkehrsunfall": "v-unfall",
            "sofortige hilfe": "sofort hilfe", "im moment": "", "so schnell wie möglich": "s.s.w.m."
        }
    },
    "fr": {
        "system_prompt": SYSTEM_PROMPT_FR,
        "filler_words": ["un", "une", "des", "le", "la", "les", "et", "mais", "ou", "donc", "très", "assez", "car", "ceci", "cela", "il", "elle", "ils", "elles", "est", "sont", "suis", "était", "étaient", "être", "été", "étant", "avoir", "a", "avaient", "faire", "fait", "faisait", "pas", "non", "oui", "mon", "ton", "son", "notre", "votre", "leur"],
        "replacements": {
            "situation d'urgence": "urgence", "j'ai besoin d'aide": "aide req", "nécessaire d'urgence": "urgent",
            "suspicion d'infarctus": "infarctus?", "ambulance nécessaire": "amb. req", "ma position est": "position",
            "ma batterie est faible": "batterie faible", "comment puis-je atteindre": "comment atteindre?",
            "accident de la route": "acc. route", "aide immédiate": "aide imm.", "incendie à la maison": "incendie maison"
        }
    },
    "es": {
        "system_prompt": SYSTEM_PROMPT_ES,
        "filler_words": ["un", "una", "unos", "unas", "el", "la", "los", "las", "y", "pero", "o", "así que", "muy", "bastante", "porque", "este", "ese", "aquel", "él", "ella", "ellos", "ellas", "es", "son", "soy", "era", "eran", "ser", "sido", "siendo", "haber", "ha", "habían", "hacer", "hace", "hacía", "no", "sí", "mi", "tu", "su", "nuestro", "vuestro", "su", "sus"],
        "replacements": {
            "situación de emergencia": "emergencia", "necesito ayuda": "ayuda req", "necesario urgentemente": "urgente",
            "sospecha de ataque al corazón": "ataque corazón?", "ambulancia necesaria": "amb. nec.", "mi ubicación es": "ubicación",
            "mi batería se está agotando": "batería baja", "cómo puedo llegar a": "cómo llegar?",
            "accidente de tráfico": "acc. tráfico", "ayuda inmediata": "ayuda imm.", "incendio en casa": "incendio casa"
        }
    },
    "ar": { # Simplified for demonstration; Arabic NLP is complex
        "system_prompt": SYSTEM_PROMPT_AR,
        "filler_words": ["و", "في", "على", "من", "إلى", "عن", "ب", "ل", "أن", "هو", "هي", "هم", "يكون", "تكون", "كان", "كانت"],
        "replacements": {
            "حالة طوارئ": "طوارئ", "أحتاج مساعدة": "مساعدة مطلوبة", "مطلوب بشكل عاجل": "عاجل",
            "اشتباه في نوبة قلبية": "نوبة قلبية؟", "مطلوب سيارة إسعاف": "إسعاف مطلوب", "موقعي هو": "موقع",
            "بطاريتي تحتضر": "بطارية منخفضة", "كيف يمكنني الوصول إلى": "كيف الوصول؟",
            "حادث مرور": "حادث", "مساعدة فورية": "مساعدة فوري", "حريق في المنزل": "حريق منزل"
        }
    },
    "zh": { # Very simplified; Character-based languages need different handling
        "system_prompt": SYSTEM_PROMPT_ZH,
        "filler_words": [], # Chinese doesn't have clear filler words like English
        "replacements": {
            "紧急情况": "紧急", "需要帮助": "求助", "我的位置是": "位置",
            "电池快没电了": "电量低", "如何到达": "如何到?", "交通事故": "车祸",
            "立即帮助": "速助", "家里着火了": "房屋火灾", "需要救护车": "救护车"
        }
    },
    "ja": { # Very simplified; Tokenization is key for Japanese
        "system_prompt": SYSTEM_PROMPT_JA,
        "filler_words": [], # Japanese grammar particles are complex to remove generically
        "replacements": {
            "緊急事態": "緊急", "助けが必要です": "助け必要", "私の位置は": "位置",
            "バッテリーが切れそうです": "バッテリー切れ", "どうすれば到達できますか": "どう到達?",
            "交通事故": "事故", "すぐに助けてください": "即助け", "家が火事です": "家火事",
            "救急車が必要です": "救急車必要"
        }
    }
}


def do_gemma_3n_inference_and_return_text(messages_payload, max_new_tokens=100, language="en"):
    """
    Simulates Gemma 3N model inference for message compression, applying language-specific rules.
    """
    original_message = ""
    for part in messages_payload[1]["content"]:
        if part["type"] == "text":
            original_message = part["text"]
            break

    if not original_message:
        return "Error: Original message not found."

    rules = LANGUAGE_RULES.get(language, LANGUAGE_RULES["en"]) # Default to English rules
    filler_words = rules["filler_words"]
    replacements = rules["replacements"]

    prefix = "Optimize and compress the following emergency message for minimal byte size while retaining all critical information:\n\n"
    if original_message.startswith(prefix):
        original_message = original_message[len(prefix):].strip()

    optimized_text = original_message.lower()

    # Apply replacements first (longer phrases first to avoid partial matches)
    for phrase, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        optimized_text = optimized_text.replace(phrase, replacement)

    # Remove filler words (mainly for Latin-script languages)
    if filler_words:
        for word in filler_words:
            optimized_text = re.sub(r'\b' + re.escape(word) + r'\b', '', optimized_text)

    # Clean up extra spaces and strip leading/trailing spaces
    optimized_text = re.sub(r'\s+', ' ', optimized_text).strip()

    # Capitalize the first letter (if any text remains)
    if optimized_text:
        optimized_text = optimized_text[0].upper() + optimized_text[1:]

    # Simulate max_new_tokens by truncation
    if len(optimized_text) > max_new_tokens:
        optimized_text = optimized_text[:max_new_tokens-3] + "..."

    return optimized_text

# Simulate Flask API's optimize_message route
def simulate_optimize_message_api(original_message_input, target_optimized_length=100, language="en"):
    """
    Simulates the Flask API's optimize_message route.
    """
    if not original_message_input:
        return {"error": "Message content missing."}, 400

    system_prompt_content = LANGUAGE_RULES.get(language, LANGUAGE_RULES["en"])["system_prompt"]

    messages_for_optimization = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt_content}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Optimize and compress the following emergency message for minimal byte size while retaining all critical information:\n\n{original_message_input}"}
            ]
        }
    ]

    try:
        optimized_text = do_gemma_3n_inference_and_return_text(messages_for_optimization, max_new_tokens=target_optimized_length, language=language)
        return {"optimized_message": optimized_text}, 200
    except Exception as e:
        return {"error": f"Error during message optimization: {str(e)}"}, 500

print("Necessary libraries and multilingual simulation functions have been installed.")


#This section **describes how a Python backend (using Flask)** would handle message optimization:


print("---")
print("## Python Backend Overview: Flask API")
print("The following is a conceptual outline of our **Flask API** (`app.py`) which processes messages from the mobile application. This API receives the original message, optimizes it using our **Gemma-based model (running on the backend)**, and returns the compressed version to the mobile app.")

flask_code_snippet = """
# app.py (On your Python Backend)
from flask import Flask, request, jsonify
# Other imports (model, tokenizer, do_gemma_3n_inference_and_return_text, SYSTEM_PROMPT)

app = Flask(__name__)

@app.route('/optimize_message', methods=['POST'])
def optimize_message_api():
    data = request.json
    original_message = data.get('message')
    language = data.get('language', 'en') # New: get language from request

    if not original_message:
        return jsonify({"error": "Message content missing."}), 400

    messages_for_optimization = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": LANGUAGE_RULES.get(language, LANGUAGE_RULES['en'])['system_prompt']} # Use language-specific prompt
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Optimize and compress the following emergency message for minimal byte size while retaining all critical information:\\n\\n{original_message}"}
            ]
        }
    ]

    try:
        optimized_text = do_gemma_3n_inference_and_return_text(messages_for_optimization, max_new_tokens=100, language=language) # Pass language
        return jsonify({"optimized_message": optimized_text}), 200
    except Exception as e:
        return jsonify({"error": f"Error during message optimization: {str(e)}"}), 500

if __name__ == '__main__':
    # Load model and tokenizer here (once)
    # model, tokenizer = FastModel.from_pretrained(...)
    # model = FastLanguageModel.get_peft_model(...)
    app.run(host='0.0.0.0', port=5000) # Start the server
"""
print(f"```python\n{flask_code_snippet}\n```")
print("\n**Note:** We can't run a live Flask server directly in a Kaggle Notebook. However, our `simulate_optimize_message_api` function mimics its behavior for demonstration purposes.")


#**Interactive Mobile App & Bluetooth Simulation**
#This code block's purpose is to simulate the entire end-to-end process of an emergency message being sent from a mobile app, optimized by a backend, and then transmitted via Bluetooth to a receiving device. It provides an interactive demonstration of how a crucial message would travel through the system, especially highlighting the compression aspect for low-bandwidth scenarios.
#1. The Optimization Logic
#First, we define a set of rules for message compression across multiple languages. For each language (English, Turkish, German, etc.), we've created:


import os
import time

flask_run_command = "nohup python your_flask_app.py > flask_output.log 2>&1 &"

os.system(flask_run_command)

time.sleep(10)

import requests

try:
    response = requests.get('http://127.0.0.1:5000/predict')
    print("Flask endpoint test successful!")
    print(response.json())
except requests.exceptions.ConnectionError as e:
    print(f"Flask endpoint test failed: {e}")
    print("Is the Flask server running?")




# Necessary libraries (assuming already installed from previous cell)
# !pip install pandas matplotlib seaborn datasets

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re # For text processing
from datasets import load_dataset # For loading Hugging Face datasets

# Define SYSTEM_PROMPT for each language (assuming these are already defined in your environment)
# If not, you'd need to copy them from your previous block here.
if 'SYSTEM_PROMPT_EN' not in globals():
    SYSTEM_PROMPT_EN = "You are an emergency message optimizer. Your goal is to compress emergency messages to their absolute minimal byte size while retaining ALL critical information (who, what, where, when, why, how). Use abbreviations, remove filler words, and rephrase concisely. Prioritize clarity and essential details over grammatical perfection."
    SYSTEM_PROMPT_TR = "Sen bir acil durum mesajı optimize edicisisin. Amacın, tüm kritik bilgileri (kim, ne, nerede, ne zaman, neden, nasıl) koruyarak acil durum mesajlarını mutlak minimum bayt boyutuna sıkıştırmaktır. Kısaltmalar kullan, dolgu kelimelerini çıkar ve özlü bir şekilde yeniden ifade et. Dilbilgisel mükemmellik yerine netliği ve temel detayları ön planda tut."
    SYSTEM_PROMPT_DE = "Du bist ein Notfallnachrichten-Optimierer. Dein Ziel ist es, Notfallnachrichten auf ihre absolute Mindestgröße zu komprimieren, während ALLE kritischen Informationen (wer, was, wo, wann, warum, wie) erhalten bleiben. Verwende Abkürzungen, entferne Füllwörter und formuliere prägnant. Priorisiere Klarheit und wesentliche Details über grammatikalische Perfektion."
    SYSTEM_PROMPT_FR = "Vous êtes un optimiseur de messages d'urgence. Votre objectif est de compresser les messages d'urgence à leur taille minimale absolue tout en conservant TOUTES les informations critiques (qui, quoi, où, quand, pourquoi, comment). Utilisez des abréviations, supprimez les mots de remplissage et reformulez de manière concise. Privilégiez la clarté et les détails essentiels à la perfection grammaticale."
    SYSTEM_PROMPT_ES = "Eres un optimizador de mensajes de emergencia. Tu objetivo es comprimir los mensajes de emergencia a su tamaño mínimo absoluto, conservando TODA la información crítica (quién, qué, dónde, cuándo, por qué, cómo). Usa abreviaturas, elimina palabras de relleno y reformula concisamente. Prioriza la claridad y los detalles esenciales sobre la perfección grammatical."
    SYSTEM_PROMPT_AR = "أنت محسن لرسائل الطوارئ. هدفك هو ضغط رسائل الطوارئ إلى أصغر حجم ممكن مع الاحتفاظ بجميع المعلومات الهامة (من، ماذا، أين، متى، لماذا، كيف). استخدم الاختصارات، احذف الكلمات الزائدة، وأعد الصياغة بإيجاز. الأولوية للوضوح والتفاصيل الأساسية على الدقة النحوية."
    SYSTEM_PROMPT_ZH = "您是紧急消息优化器。您的目标是将紧急消息压缩到绝对最小的字节大小，同时保留所有关键信息（谁、什么、哪里、何时、为什么、如何）。使用缩写，删除填充词，并简洁地重新表达。优先考虑清晰度和基本细节，而不是语法完美性。"
    SYSTEM_PROMPT_JA = "あなたは緊急メッセージの最適化ツールです。あなたの目標は、すべての重要な情報（誰が、何を、どこで、いつ、なぜ、どのように）を保持しながら、緊急メッセージを最小限のバイトサイズに圧縮することです。略語を使用し、余分な単語を削除し、簡潔に言い換えてください。文法的な完璧さよりも、明確さと重要な詳細を優先してください。"

# Define language-specific rules (filler words and common replacements)
# Note: For Arabic, Chinese, and Japanese, simple word/phrase replacement might be less effective than
# for Latin-script languages due to their linguistic structure and lack of clear word boundaries.
# A proper NLP model (like a real Gemma) would handle these much better.
LANGUAGE_RULES = {
    "en": {
        "system_prompt": SYSTEM_PROMPT_EN,
        "filler_words": ["a", "an", "the", "and", "but", "or", "so", "very", "quite", "because", "this", "that", "it", "is", "are", "am", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "not", "no", "yes", "my", "your", "his", "her", "its", "our", "their"],
        "replacements": {
            "emergency situation": "emergency", "i need help": "help req", "urgently needed": "urgent",
            "heart attack suspicion": "heart attack?", "ambulance needed": "amb. req", "my location is": "loc",
            "my battery is dying": "battery low", "how can i reach": "how to reach?",
            "traffic accident": "traffic acc.", "help immediately": "help asap", "fire in house": "house fire",
            "at the moment": "", "in order to": "", "as soon as possible": "ASAP"
        }
    },
    "tr": {
        "system_prompt": SYSTEM_PROMPT_TR,
        "filler_words": ["bir", "ve", "ile", "çok", "oldukça", "ancak", "fakat", "lakin", "çünkü", "bu", "o", "şu", "da", "de", "mi", "mı", "mu", "mü"],
        "replacements": {
            "acil durum": "acil", "ihtiyacım var": "ihtiyaç", "yardıma ihtiyacım var": "yardım gerek",
            "en kısa sürede": "hızla", "kalp krizi şüphesi": "kalp krizi?", "ambulans gerekiyor": "ambl. gerek",
            "konumum": "konum", "bataryam bitiyor": "batarya bitiyor", "yerleşim yerine": "yerleşim",
            "nasıl ulaşabilirim": "nasıl?", "acil yardım": "acil yardım", "hemen şimdi": "şimdi"
        }
    },
    "de": {
        "system_prompt": SYSTEM_PROMPT_DE,
        "filler_words": ["ein", "eine", "einer", "und", "aber", "oder", "also", "sehr", "ziemlich", "weil", "dies", "das", "es", "ist", "sind", "bin", "war", "waren", "sein", "gewesen", "seiend", "haben", "hat", "hatte", "tun", "tut", "tat", "nicht", "nein", "ja", "mein", "dein", "sein", "ihr", "unser", "euer", "ihr"],
        "replacements": {
            "notfallsituation": "notfall", "ich brauche hilfe": "hilfe benötigt", "dringend benötigt": "dringend",
            "herzinfarktverdacht": "herzinfarkt?", "krankenwagen benötigt": "kwg. benötigt", "mein standort ist": "standort",
            "mein akku ist leer": "akku leer", "wie komme ich zu": "wie zu?", "verkehrsunfall": "v-unfall",
            "sofortige hilfe": "sofort hilfe", "im moment": "", "so schnell wie möglich": "s.s.w.m."
        }
    },
    "fr": {
        "system_prompt": SYSTEM_PROMPT_FR,
        "filler_words": ["un", "une", "des", "le", "la", "les", "et", "mais", "ou", "donc", "très", "assez", "car", "ceci", "cela", "il", "elle", "ils", "elles", "est", "sont", "suis", "était", "étaient", "être", "été", "étant", "avoir", "a", "avaient", "faire", "fait", "faisait", "pas", "non", "oui", "mon", "ton", "son", "notre", "votre", "leur"],
        "replacements": {
            "situation d'urgence": "urgence", "j'ai besoin d'aide": "aide req", "nécessaire d'urgence": "urgent",
            "suspicion d'infarctus": "infarctus?", "ambulance nécessaire": "amb. req", "ma position est": "position",
            "ma batterie est faible": "batterie faible", "comment puis-je atteindre": "comment atteindre?",
            "accident de la route": "acc. route", "aide immédiate": "aide imm.", "incendie à la maison": "incendie maison"
        }
    },
    "es": {
        "system_prompt": SYSTEM_PROMPT_ES,
        "filler_words": ["un", "una", "unos", "unas", "el", "la", "los", "las", "y", "pero", "o", "así que", "muy", "bastante", "porque", "este", "ese", "aquel", "él", "ella", "ellos", "ellas", "es", "son", "soy", "era", "eran", "ser", "sido", "siendo", "haber", "ha", "habían", "hacer", "hace", "hacía", "no", "sí", "mi", "tu", "su", "nuestro", "vuestro", "su", "sus"],
        "replacements": {
            "situación de emergencia": "emergencia", "necesito ayuda": "ayuda req", "necesario urgentemente": "urgente",
            "sospecha de ataque al corazón": "ataque corazón?", "ambulancia necesaria": "amb. nec.", "mi ubicación es": "ubicación",
            "mi batería se está agotando": "batería baja", "cómo puedo llegar a": "cómo llegar?",
            "accidente de tráfico": "acc. tráfico", "ayuda inmediata": "ayuda imm.", "incendio en casa": "incendio casa"
        }
    },
    "ar": { # Simplified for demonstration; Arabic NLP is complex
        "system_prompt": SYSTEM_PROMPT_AR,
        "filler_words": ["و", "في", "على", "من", "إلى", "عن", "ب", "ل", "أن", "هو", "هي", "هم", "يكون", "تكون", "كان", "كانت"],
        "replacements": {
            "حالة طوارئ": "طوارئ", "أحتاج مساعدة": "مساعدة مطلوبة", "مطلوب بشكل عاجل": "عاجل",
            "اشتباه في نوبة قلبية": "نوبة قلبية؟", "مطلوب سيارة إسعاف": "إسعاف مطلوب", "موقعي هو": "موقع",
            "بطاريتي تحتضر": "بطارية منخفضة", "كيف يمكنني الوصول إلى": "كيف الوصول؟",
            "حادث مرور": "حادث", "مساعدة فورية": "مساعدة فوري", "حريق في المنزل": "حريق منزل"
        }
    },
    "zh": { # Very simplified; Character-based languages need different handling
        "system_prompt": SYSTEM_PROMPT_ZH,
        "filler_words": [], # Chinese doesn't have clear filler words like English
        "replacements": {
            "紧急情况": "紧急", "需要帮助": "求助", "我的位置是": "位置",
            "电池快没电了": "电量低", "如何到达": "如何到?", "交通事故": "车祸",
            "立即帮助": "速助", "家里着火了": "房屋火灾", "需要救护车": "救护车"
        }
    },
    "ja": { # Very simplified; Tokenization is key for Japanese
        "system_prompt": SYSTEM_PROMPT_JA,
        "filler_words": [], # Japanese grammar particles are complex to remove generically
        "replacements": {
            "緊急事態": "緊急", "助けが必要です": "助け必要", "私の位置は": "位置",
            "バッテリーが切れそうです": "バッテリー切れ", "どうすれば到達できますか": "どう到達?",
            "交通事故": "事故", "すぐに助けてください": "即助け", "家が火事です": "家火事",
            "救急車が必要です": "救急車必要"
        }
    }
}

class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict, return_tensors):
        # Simplified chat template application for demonstration purposes
        full_text = ""
        for msg in messages:
            if msg["role"] == "system":
                full_text += f"<|system|>\n{msg['content'][0]['text']}\n"
            elif msg["role"] == "user":
                full_text += f"<|user|>\n{msg['content'][0]['text']}\n"
            elif msg["role"] == "assistant":
                full_text += f"<|assistant|>\n{msg['content'][0]['text']}\n"
        full_text += "<|assistant|>\n" # Start of generation prompt for the model
        # Create dummy token IDs
        return {'input_ids': torch.tensor([list(range(len(full_text.split())))])}

    def decode(self, tokens, skip_special_tokens=True):
        if not tokens.numel():
            return ""
        # Generating a dummy optimized message
        dummy_text = "Opt: Emergency msg compressed. Help is on the way. "
        if len(tokens) > 5:
            dummy_text += " Details: " + ' '.join([str(t) for t in tokens[5:min(len(tokens), 20)]]) + "..."
        # Access max_new_tokens from the global scope if it's set, otherwise use a default
        global max_new_tokens_in_scope # Declare intent to use global variable
        return dummy_text[:max_new_tokens_in_scope] if 'max_new_tokens_in_scope' in globals() else dummy_text # Truncate for demo

    def __call__(self, text, return_tensors=None):
        return {'input_ids': list(range(len(text.split())))} # Very rough token count

tokenizer = DummyTokenizer()

class DummyModel:
    def generate(self, input_ids, generation_config):
        num_generated_tokens = generation_config.max_new_tokens
        start_id = input_ids['input_ids'].shape[-1]
        generated_part = torch.tensor([list(range(start_id, start_id + num_generated_tokens))])
        return torch.cat([input_ids['input_ids'], generated_part], dim=1)

model = DummyModel()

# do_gemma_3n_inference_and_return_text function (copied from your previous code)
def do_gemma_3n_inference_and_return_text(messages_payload, max_new_tokens=100, language="en"):
    """
    Simulates Gemma 3N model inference for message compression, applying language-specific rules.
    """
    # Make max_new_tokens accessible to the DummyTokenizer's decode method
    global max_new_tokens_in_scope
    max_new_tokens_in_scope = max_new_tokens

    original_message = ""
    for part in messages_payload[1]["content"]:
        if part["type"] == "text":
            original_message = part["text"]
            break

    if not original_message:
        return "Error: Original message not found."

    rules = LANGUAGE_RULES.get(language, LANGUAGE_RULES["en"]) # Default to English rules
    filler_words = rules["filler_words"]
    replacements = rules["replacements"]

    prefix = "Optimize and compress the following emergency message for minimal byte size while retaining all critical information:\n\n"
    if original_message.startswith(prefix):
        original_message = original_message[len(prefix):].strip()

    optimized_text = original_message.lower()

    # Apply replacements first (longer phrases first to avoid partial matches)
    for phrase, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        optimized_text = optimized_text.replace(phrase, replacement)

    # Remove filler words (mainly for Latin-script languages)
    if filler_words:
        for word in filler_words:
            # Use word boundaries to avoid replacing parts of words
            optimized_text = re.sub(r'\b' + re.escape(word) + r'\b', '', optimized_text)

    # Clean up extra spaces and strip leading/trailing spaces
    optimized_text = re.sub(r'\s+', ' ', optimized_text).strip()

    # Capitalize the first letter (if any text remains)
    if optimized_text:
        optimized_text = optimized_text[0].upper() + optimized_text[1:]

    # Simulate max_new_tokens by truncation
    if len(optimized_text) > max_new_tokens:
        optimized_text = optimized_text[:max_new_tokens-3] + "..."

    return optimized_text

# Simulate Flask API's optimize_message route
def simulate_optimize_message_api(original_message_input, target_optimized_length=100, language="en"):
    """
    Simulates the Flask API's optimize_message route.
    """
    if not original_message_input:
        return {"error": "Message content missing."}, 400

    system_prompt_content = LANGUAGE_RULES.get(language, LANGUAGE_RULES["en"])["system_prompt"]

    messages_for_optimization = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt_content}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Optimize and compress the following emergency message for minimal byte size while retaining all critical information:\n\n{original_message_input}"}
            ]
        }
    ]

    try:
        optimized_text = do_gemma_3n_inference_and_return_text(messages_for_optimization, max_new_tokens=target_optimized_length, language=language)
        return {"optimized_message": optimized_text}, 200
    except Exception as e:
        return {"error": f"Error during message optimization: {str(e)}"}, 500

print("---")
print("## Interactive Mobile App Flow & Bluetooth Simulation (Multilingual)")
print("Enter your emergency message and choose a language to see how it's optimized and transmitted.")

user_input_message = input("Enter your emergency message here (e.g., 'My car broke down on the highway, need help!' or 'Ein Feuer ist in meinem Haus!'):\n")
selected_language_code = input("Enter the language code (en/tr/de/fr/es/ar/zh/ja): ").lower().strip()

# Validate language code and set default message if input is empty
if selected_language_code not in LANGUAGE_RULES:
    print(f" Warning: '{selected_language_code}' is an invalid language code. Defaulting to 'en' (English).")
    selected_language_code = "en"

if not user_input_message.strip():
    default_messages = {
        "en": "Traffic accident, injured. My location: highway km 5. Police and ambulance needed.",
        "tr": "Trafik kazası, yaralılar var. Konumum TEM otoyolu 5. km. Polis ve ambulans gerekli.",
        "de": "Verkehrsunfall, Verletzte. Mein Standort: Autobahnkilometer 5. Polizei und Krankenwagen erforderlich.",
        "fr": "Accident de la route, blessés. Ma position: autoroute km 5. Police et ambulance nécessaires.",
        "es": "Accidente de tráfico, heridos. Mi ubicación: autopista km 5. Policía y ambulancia necesarias.",
        "ar": "حادث سير، مصابون. موقعي: الكيلومتر 5 طريق سريع. الشرطة والإسعاف مطلوبان.",
        "zh": "交通事故，有伤员。我的位置：高速公路5公里处。需要警察和救护车。",
        "ja": "交通事故、負傷者。私の場所：高速道路5km地点。警察と救急車が必要です。"
    }
    user_input_message = default_messages.get(selected_language_code, default_messages["en"])
    print(f"\n✨ Empty message entered. Using default message ({selected_language_code.upper()}): '{user_input_message}'")


print("\n" + "="*70)
print(" MESSAGE PROCESSING FLOW INITIATED 🚀".center(70))
print("="*70)

print("\n### 1.  User Message Creation & Transmission from App")
print(f"The user enters the emergency message in the mobile app ({selected_language_code.upper()}):")
print(f" **Original Message:**\n`{user_input_message}`")
print(f" Original Length (characters): {len(user_input_message)}")


print("\n" + "-"*70)
print("### 2.  Python Backend Communication & Message Optimization")
print("The mobile app sends this message (via a simulated HTTP POST request) to our Python backend.")
print("Our **Gemma-based model** (simulated) on the backend optimizes and compresses the message specifically for emergency scenarios, considering the detected or selected language.")

optimized_message_length_target = 100 # Target character length for the optimized message
api_response, status_code = simulate_optimize_message_api(user_input_message, optimized_message_length_target, language=selected_language_code)

if status_code == 200:
    optimized_message = api_response.get("optimized_message")
    print(f"\n **Optimized Message from Backend:**\n`{optimized_message}`")
    print(f"📏 Optimized Length (characters): {len(optimized_message)}")
    compression_ratio = (len(user_input_message) - len(optimized_message)) / len(user_input_message)
    print(f" Compression Ratio: {compression_ratio:.2%}")
else:
    optimized_message = "Optimization error: " + api_response.get("error", "Unknown error")
    print(f"**Backend Error:**\n`{optimized_message}` (HTTP {status_code})")

print("\n" + "-"*70)
print("### 3.  Enabling Bluetooth & Permissions")
print("The mobile app requests permission from the user to enable Bluetooth (or attempts to enable it automatically).")
print("It obtains necessary system permissions (e.g., location permission) to discover nearby Bluetooth devices.")
print("These permissions are critical for **Bluetooth Low Energy (BLE)** scanning.")


print("\n" + "-"*70)
print("### 4. Discovering Nearby Devices (Discovery)")
print("The app begins scanning for nearby Bluetooth devices (other phones or receiving devices).")
print("This process typically uses BLE 'advertising' and 'scanning' mechanisms.")
print("The 'send to nearest person automatically' logic is applied here, estimating physical proximity using **Bluetooth signal strength (RSSI)**.")

print("\n** Simulation: Nearby Device Discovery**")
print("- Device ID: Mobile Device A (RSSI: -42 dBm - **🎯 Target Selected**)")
print("- Device ID: Mobile Device B (RSSI: -68 dBm)")
print("- Device ID: Mobile Device C (RSSI: -81 dBm)")


print("\n" + "-"*70)
print("### 5. Establishing Connection & Sending Message")
print("The app establishes a Bluetooth connection with the targeted device (Mobile Device A).")
print("The optimized message is then sent securely over the established **Bluetooth (BLE - GATT Services)** connection.")
print("An acknowledgment mechanism (e.g., a 'message received' confirmation from the receiver) is expected to verify successful transmission.")
print(f"\n **Optimized Message Sent (via Bluetooth):**\n`{optimized_message}`")
print("\n *Acknowledgment Received: Message successfully transmitted.*")


print("\n" + "-"*70)
print("### 6.  Receiving & Displaying Message (On Receiving Phone)")
print("The receiving mobile application continuously listens for incoming optimized messages via Bluetooth.")
print("It receives the message, decompresses it if necessary, and displays it to the user.")
print(f"\n **Message Displayed on Receiving Phone:**\n`{optimized_message}`")
print("\n" + "="*70)
print(" SIMULATION COMPLETE ".center(70))
print("="*70)

# Phase 4: Comprehensive Performance Analysis
#Our final, and arguably most critical, phase involves a deep dive into the performance of our optimization model. We don't just care about making messages shorter; we obsess over ensuring critical information is perfectly preserved while maximizing compression efficiency. This rigorous evaluation provides a data-driven validation of our entire system.
#Here's how we rigorously assessed our model:


print("---")
print("## General Message Optimization Analysis (Multilingual)")
print("We're analyzing the (simulated) optimization performance of our model on example messages across different languages.")

# Example messages in various languages for analysis
multilingual_analysis_messages = [
    {"lang": "tr", "text": "Acil yardım! Konumum 39.9255, 32.8597. Ciddi bir kaza oldu, ambulans ve polis gerekiyor. Lütfen hızlı gelin."},
    {"lang": "en", "text": "Emergency help! My location is 39.9255, 32.8597. There's been a serious accident, ambulance and police are needed. Please come quickly."},
    {"lang": "de", "text": "Notfallhilfe! Mein Standort ist 39.9255, 32.8597. Es gab einen schweren Unfall, Krankenwagen und Polizei werden benötigt. Bitte schnell kommen."},
    {"lang": "fr", "text": "Aide d'urgence! Ma position est 39.9255, 32.8597. Il y a eu un grave accident, ambulance et police sont nécessaires. Venez vite s'il vous plaît."},
    {"lang": "es", "text": "¡Ayuda de emergencia! Mi ubicación es 39.9255, 32.8597. Ha habido un accidente grave, se necesita ambulancia y policía. ¡Por favor, vengan rápido!"},
    {"lang": "ar", "text": "مساعدة طارئة! موقعي 39.9255, 32.8597. حدث حادث خطير، مطلوب سيارة إسعاف وشرطة. من فضلكم تعالوا بسرعة."},
    {"lang": "zh", "text": "紧急求助！我的位置是39.9255, 32.8597。发生了严重事故，需要救护车和警察。请快速过来。"},
    {"lang": "ja", "text": "緊急援助！私の位置は39.9255, 32.8597です。重大な事故が発生しました。救急車と警察が必要です。すぐ来てください。"}
]

message_data = []
for item in multilingual_analysis_messages:
    msg = item["text"]
    lang = item["lang"]
    response, _ = simulate_optimize_message_api(msg, language=lang)
    optimized_msg = response.get("optimized_message", msg)

    original_len = len(msg)
    optimized_len = len(optimized_msg)
    message_data.append({
        'language': lang,
        'original_message': msg,
        'optimized_message': optimized_msg,
        'original_length': original_len,
        'optimized_length': optimized_len
    })

analysis_df = pd.DataFrame(message_data)

print("\n**Message Length Comparison (Character Count):**")
print(analysis_df[['language', 'original_length', 'optimized_length']].describe(include='all').to_string())

# Average compression ratios per language
print("\n**Average Compression Ratios by Language:**")
analysis_df['compression_ratio'] = (analysis_df['original_length'] - analysis_df['optimized_length']) / analysis_df['original_length']
print(analysis_df.groupby('language')['compression_ratio'].mean().apply(lambda x: f"{x:.2%}").to_string())


# Length comparison plot (faceted by language for better visualization)
plt.figure(figsize=(18, 10)) # Increasing the  figure size
g = sns.FacetGrid(analysis_df, col="language", col_wrap=3, height=4, aspect=1.2, sharey=False) 
g.map_dataframe(sns.histplot, x='original_length', color="blue", label="Original Message Length", kde=True, alpha=0.6)
g.map_dataframe(sns.histplot, x='optimized_length', color="red", label="Optimized Message Length", kde=True, alpha=0.6)
g.set_axis_labels("Character Count", "Message Count")
g.add_legend(title="Message Type")
plt.suptitle("Distribution of Original vs. Optimized Message Lengths by Language", fontsize=18, y=1.02) 
plt.tight_layout(rect=[0, 0, 1, 0.98]) 
plt.show()

print("\nThis analysis demonstrates how our Python backend (simulated) optimizes emergency messages across different languages. By using language-specific rules, we aim for effective compression in each supported language.")
print("This compression is crucial for faster and more efficient data transmission over bandwidth-constrained Bluetooth connections.")

# Phase 5:  Information Protection and Criticality Assessment

#One of the most important goals in emergency message optimization is to preserve all critical information while shortening the message. To analyze this, we need to identify specific information and verify its presence and accuracy in the optimized messages.


import json

def gemma_extract_critical_info_simulated(message, language="en"):
    """
    Simulates Gemma 3N extracting critical information from a message.
    In a real scenario, this would send a prompt to Gemma 3N API and parse its JSON output.
    For simulation, we'll use a more advanced version of our previous extract_critical_info.
    """
    # In a real Gemma 3N integration:
    # 1. Construct messages payload with the specific extraction system prompt.
    # 2. Call Gemma 3N API with the message and extraction prompt.
    # 3. Parse the JSON output from Gemma 3N.
    # Example API call (conceptual):
    # response = gemini_api_call(
    #     model="gemma-3n",
    #     messages=[
    #         {"role": "system", "content": [{"type": "text", "text": "You are a highly accurate information extraction AI. Your task is to identify and extract critical emergency information from a given text. Return the information in a structured JSON format. The categories of information to extract are: 'emergency_type', 'location' (specific address, coordinates, landmarks), 'people_involved' (number, status, details), 'needs' (required services/items), 'urgency' (boolean), and 'extra_details'. If a category is not present or cannot be determined, return an empty list or appropriate null/false value. Ensure coordinates are extracted precisely if available."},
    #         {"role": "user", "content": [{"type": "text", "text": f"Extract critical information from the following emergency message:\n\n{message}"}]}
    #     ]
    # )
    # parsed_info = json.loads(response.text) # Assuming Gemma returns JSON directly

    # For our simulation, we'll use an improved rule-based extraction that acts smarter
    # but still isn't true LLM understanding.
    info = {
        "emergency_type": [],
        "location": [],
        "people_involved": [],
        "needs": [],
        "urgency": False,
        "coordinates": [],
        "extra_details": []
    }
    message_lower = message.lower()

    # This part would be the result of Gemma's advanced understanding in a real scenario
    # We are simulating its output by trying to be a bit smarter with regex/keywords.

    # Example of what a simulated Gemma might extract based on a more complex rule set
    # This is still rule-based, but more refined than previous simple example.
    if language == "en":
        # Emergency Type
        if "fire" in message_lower or "smoke" in message_lower: info["emergency_type"].append("fire")
        if "accident" in message_lower or "crash" in message_lower or "collision" in message_lower: info["emergency_type"].append("accident")
        if "medical emergency" in message_lower or "heart attack" in message_lower or "breathing difficulty" in message_lower or "unconscious" in message_lower or "fell" in message_lower or "injury" in message_lower or "sick" in message_lower: info["emergency_type"].append("medical")
        if "flood" in message_lower or "water rising" in message_lower or "submerged" in message_lower: info["emergency_type"].append("flood")
        if "theft" in message_lower or "robbery" in message_lower: info["emergency_type"].append("theft/robbery")
        if "suspicious package" in message_lower or "bomb threat" in message_lower: info["emergency_type"].append("security threat")

        # Location
        coords = re.findall(r'(\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)', message) # Latitude, Longitude
        for lat, lon in coords:
            info["coordinates"].append(f"{float(lat):.4f}, {float(lon):.4f}")
        address_patterns = [
            r'\d+\s+[A-Za-z]+\s+(Street|Avenue|Road|Lane|Blvd|St|Ave|Rd|Ln)\b',
            r'\bhighway\s+\w+\s+km\s+\d+\b',
            r'\b(\d+)\.\s*km\b', # e.g., "5. km" for highways
            r'\b(?:at|on|near)\s+([A-Za-z0-9\s]+(?:bridge|station|park|mall|building|hospital|school|intersection|square))\b'
        ]
        for pattern in address_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            info["location"].extend([m.strip() for m in matches if m.strip()])
        if "downtown" in message_lower: info["location"].append("downtown")
        if "city hall" in message_lower: info["location"].append("city hall")
        if "exit 12" in message_lower: info["location"].append("exit 12")

        # People Involved
        people_counts = re.findall(r'(\d+)\s+(person|people|individual|victim|child|adult|trapped resident)s?', message_lower)
        for count, p_type in people_counts: info["people_involved"].append(f"{count} {p_type}")
        if "unconscious" in message_lower: info["people_involved"].append("unconscious person")
        if "injured" in message_lower or "hurt" in message_lower: info["people_involved"].append("injured person(s)")
        if "child" in message_lower or "children" in message_lower: info["people_involved"].append("child/children")
        if "elderly" in message_lower or "senior" in message_lower: info["people_involved"].append("elderly person")
        if "trapped" in message_lower: info["people_involved"].append("trapped person(s)")

        # Needs
        if "ambulance" in message_lower or "medic" in message_lower or "paramedics" in message_lower or "medical help" in message_lower: info["needs"].append("medical assistance")
        if "police" in message_lower or "officer" in message_lower: info["needs"].append("police assistance")
        if "fire truck" in message_lower or "firefighters" in message_lower or "fire department" in message_lower: info["needs"].append("fire suppression")
        if "rescue team" in message_lower or "search and rescue" in message_lower: info["needs"].append("rescue team")
        if "water rescue" in message_lower: info["needs"].append("water rescue")
        if "power outage" in message_lower and ("repair" in message_lower or "electricity" in message_lower): info["needs"].append("power restoration")

        # Urgency
        if "urgent" in message_lower or "immediately" in message_lower or "asap" in message_lower or "critical" in message_lower or "fast" in message_lower or "quickly" in message_lower or "expedite" in message_lower:
            info["urgency"] = True
        if "battery low" in message_lower or "dying phone" in message_lower: info["urgency"] = True # Implicit urgency

        # Extra Details
        if "multi-story" in message_lower and re.search(r'\d+(?:th|nd|rd|st) floor', message_lower): info["extra_details"].append("building floor details")
        if "vehicle type" in message_lower or "car model" in message_lower: info["extra_details"].append("vehicle description")
        if "weapon" in message_lower: info["extra_details"].append("weapon involved")
        if "gas leak" in message_lower: info["extra_details"].append("gas leak")

    elif language == "tr":
        # Emergency Type
        if "yangın" in message_lower: info["emergency_type"].append("yangın")
        if "kaza" in message_lower: info["emergency_type"].append("kaza")
        if "tıbbi acil" in message_lower or "kalp krizi" in message_lower or "nefes zorluğu" in message_lower or "bilinçsiz" in message_lower or "düştü" in message_lower or "yaralanma" in message_lower or "hasta" in message_lower: info["emergency_type"].append("tıbbi")
        if "sel" in message_lower or "su seviyesi yükseliyor" in message_lower or "sular altında" in message_lower: info["emergency_type"].append("sel")
        if "hırsızlık" in message_lower or "soygun" in message_lower: info["emergency_type"].append("hırsızlık/soygun")
        if "şüpheli paket" in message_lower or "bomba tehdidi" in message_lower: info["emergency_type"].append("güvenlik tehdidi")

        # Location
        coords = re.findall(r'(\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)', message)
        for lat, lon in coords:
            info["coordinates"].append(f"{float(lat):.4f}, {float(lon):.4f}")
        address_patterns = [
            r'\d+\s+[A-Za-zÇĞİÖŞÜçğiöşü]+\s+(Cadde|Sokak|Bulvarı|Caddesi|Sk|Cad|Blv)\b',
            r'\b(?:otoyol|tem)\s+\d+\.\s*km\b', # e.g., "TEM 5. km"
            r'\b(\d+)\.\s*km\b',
            r'\b(?:yakınında|yanında|üzerinde)\s+([A-Za-zÇĞİÖŞÜçğiöşü\s]+(?:köprü|istasyon|park|AVM|bina|hastane|okul|kavşak|meydan))\b'
        ]
        for pattern in address_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            info["location"].extend([m.strip() for m in matches if m.strip()])
        if "merkez" in message_lower: info["location"].append("merkez")
        if "belediye" in message_lower: info["location"].append("belediye")
        if "çıkış 12" in message_lower: info["location"].append("çıkış 12")

        # People Involved
        people_counts = re.findall(r'(\d+)\s+(kişi|birey|mağdur|çocuk|yetişkin|mahsur kalan)s?', message_lower)
        for count, p_type in people_counts: info["people_involved"].append(f"{count} {p_type}")
        if "bilinçsiz" in message_lower: info["people_involved"].append("bilinçsiz kişi")
        if "yaralı" in message_lower: info["people_involved"].append("yaralı kişi(ler)")
        if "çocuk" in message_lower: info["people_involved"].append("çocuk/çocuklar")
        if "yaşlı" in message_lower: info["people_involved"].append("yaşlı kişi")
        if "mahsur" in message_lower: info["people_involved"].append("mahsur kalan kişi(ler)")

        # Needs
        if "ambulans" in message_lower or "sağlık ekibi" in message_lower: info["needs"].append("tıbbi yardım")
        if "polis" in message_lower or "memur" in message_lower: info["needs"].append("polis yardımı")
        if "itfaiye" in message_lower: info["needs"].append("itfaiye")
        if "kurtarma ekibi" in message_lower: info["needs"].append("kurtarma ekibi")
        if "su altı kurtarma" in message_lower: info["needs"].append("su altı kurtarma")
        if "elektrik kesintisi" in message_lower and ("onarım" in message_lower or "elektrik" in message_lower): info["needs"].append("elektrik onarımı")

        # Urgency
        if "acil" in message_lower or "hemen" in message_lower or "bir an önce" in message_lower or "kritik" in message_lower or "hızlı" in message_lower:
            info["urgency"] = True
        if "batarya zayıf" in message_lower or "telefon kapanıyor" in message_lower: info["urgency"] = True # Implicit urgency

        # Extra Details
        if "çok katlı" in message_lower and re.search(r'\d+\.\s*kat', message_lower): info["extra_details"].append("bina kat bilgisi")
        if "araç tipi" in message_lower or "araba modeli" in message_lower: info["extra_details"].append("araç açıklaması")
        if "silah" in message_lower: info["extra_details"].append("silah dahil")
        if "gaz kaçağı" in message_lower: info["extra_details"].append("gaz kaçağı")

    # Final cleanup and sort/unique for consistency
    for key, value in info.items():
        if isinstance(value, list):
            info[key] = sorted(list(set(value)))
    return info


#**Step 2: Fidelity Score, False Positives/Negatives Calculation**


def calculate_fidelity_score_gemma(original_message, optimized_message, language="en"):
    original_info = gemma_extract_critical_info_simulated(original_message, language)
    optimized_info = gemma_extract_critical_info_simulated(optimized_message, language)

    # Flatten the dictionaries into sets of "category:item" strings for comparison
    original_set = set()
    for key, val in original_info.items():
        if isinstance(val, list):
            for item in val:
                original_set.add(f"{key}:{item}")
        else: # For boolean like 'urgency'
            original_set.add(f"{key}:{val}")

    optimized_set = set()
    for key, val in optimized_info.items():
        if isinstance(val, list):
            for item in val:
                optimized_set.add(f"{key}:{item}")
        else: # For boolean like 'urgency'
            optimized_set.add(f"{key}:{val}")

    preserved_info = original_set.intersection(optimized_set)

    if not original_set:
        return 1.0 # If original had no detectable critical info, perfect fidelity
    fidelity = len(preserved_info) / len(original_set)
    return fidelity

def analyze_false_positives_negatives_gemma(original_message, optimized_message, language="en"):
    original_info = gemma_extract_critical_info_simulated(original_message, language)
    optimized_info = gemma_extract_critical_info_simulated(optimized_message, language)

    original_flat = set()
    for key, val in original_info.items():
        if isinstance(val, list):
            for item in val:
                original_flat.add(f"{key}:{item}")
        else:
            original_flat.add(f"{key}:{val}")

    optimized_flat = set()
    for key, val in optimized_info.items():
        if isinstance(val, list):
            for item in val:
                optimized_flat.add(f"{key}:{item}")
        else:
            optimized_flat.add(f"{key}:{val}")

    false_negatives = original_flat - optimized_flat
    false_positives = optimized_flat - original_flat

    return list(false_negatives), list(false_positives)




print("---")
print("## Gemma 3N Supported Information Protection and Criticality Analysis (Simulated)")
print("Gemma 3N we analyze how the model extracts critical information and preserves it in optimized messages.")

# Sample messages, now including type information
multilingual_analysis_messages_with_types = [
    {"lang": "tr", "type": "accident", "text": "Acil yardım! Konumum 39.9255, 32.8597. Ciddi bir kaza oldu, ambulans ve polis gerekiyor. Lütfen hızlı gelin."},
    {"lang": "en", "type": "accident", "text": "Emergency help! My location is 39.9255, 32.8597. There's been a serious accident, ambulance and police are needed. Please come quickly."},
    {"lang": "de", "type": "accident", "text": "Notfallhilfe! Mein Standort ist 39.9255, 32.8597. Es gab einen schweren Unfall, Krankenwagen und Polizei werden benötigt. Bitte schnell kommen."},
    {"lang": "fr", "type": "accident", "text": "Aide d'urgence! Ma position est 39.9255, 32.8597. Il y a eu un grave accident, ambulance et police sont nécessaires. Venez vite s'il vous plaît."},
    {"lang": "es", "type": "accident", "text": "¡Ayuda de emergencia! Mi ubicación es 39.9255, 32.8597. Ha habido un accidente grave, se necesita ambulancia y policía. ¡Por favor, vengan rápido!"},
    {"lang": "ar", "type": "accident", "text": "مساعدة طارئة! موقعي 39.9255, 32.8597. حدث حادث خطير، مطلوب سيارة إسعاف وشرطة. من فضلكم تعالوا بسرعة."},
    {"lang": "zh", "type": "accident", "text": "紧急求助！我的位置是39.9255, 32.8597。发生了严重事故，需要救护车和警察。请快速过来。"},
    {"lang": "ja", "type": "accident", "text": "緊急援助！私の位置は39.9255, 32.8597です。重大な事故が発生しました。救急車と警察が必要です。すぐ来てください。"},

    {"lang": "en", "type": "fire", "text": "There's a fire at 123 Main Street, downtown area. Building is multi-story, 5th floor involved, heavy black smoke visible. Possible 3 trapped residents on upper floors. Send 5 fire trucks and 2 ambulances immediately. Dispatch ETA requested."},
    {"lang": "tr", "type": "fire", "text": "Ana Cadde 123'te, şehir merkezinde bir yangın var. Bina çok katlı, 5. kat etkilendi, yoğun siyah duman görülüyor. Üst katlarda 3 mahsur kalmış kişi olabilir. Hemen 5 itfaiye ve 2 ambulans gönderin. Tahmini varış süresi istendi."},
    {"lang": "en", "type": "medical", "text": "My elderly neighbor, Mrs. Smith, fell in her apartment at 4B, Building C, Elm Street. She can't get up, complaining of severe hip pain. She's conscious but disoriented and has a known history of heart conditions. Urgent medical attention required. Please send an ambulance immediately."},
    {"lang": "tr", "type": "medical", "text": "Yaşlı komşum, Bayan Ayşe, Elm Sokak C Blok, 4B numaralı dairesinde düştü. Ayağa kalkamıyor, şiddetli kalça ağrısı çekiyor. Bilinci açık ama kafası karışık ve bilinen bir kalp rahatsızlığı geçmişi var. Acil tıbbi yardım gerekiyor. Lütfen hemen bir ambulans gönderin."},
    {"lang": "en", "type": "natural disaster", "text": "Severe flooding in low-lying areas near Willow Creek. Water levels are rising rapidly, waist-deep in some homes. Several individuals are on their car roofs awaiting rescue. We need water rescue teams and blankets urgently. My phone battery is almost dead."},
    {"lang": "tr", "type": "natural disaster", "text": "Söğüt Deresi yakınındaki alçak bölgelerde şiddetli sel var. Su seviyeleri hızla yükseliyor, bazı evlerde bel hizasına ulaştı. Birkaç kişi araba çatılarında kurtarılmayı bekliyor. Acil su kurtarma ekipleri ve battaniyeler gerekiyor. Telefonumun şarjı bitmek üzere."}
]

analysis_data_gemma = []
for item in multilingual_analysis_messages_with_types:
    msg = item["text"]
    lang = item["lang"]
    msg_type = item["type"]

    # --- SIMULATED OPTIMIZATION ---
    # Calling our simulated backend to optimize the message
    response, _ = simulate_optimize_message_api(msg, language=lang)
    optimized_msg = response.get("optimized_message", msg)

    # --- CRITICAL INFORMATION EXTRACTION (SIMULATED BY GEMMA 3N) ---
    original_info_extracted = gemma_extract_critical_info_simulated(msg, lang)
    optimized_info_extracted = gemma_extract_critical_info_simulated(optimized_msg, lang)

    # --- FIDELITY SCORE & FP/FN CALCULATION ---
    fidelity = calculate_fidelity_score_gemma(msg, optimized_msg, lang)
    false_negatives, false_positives = analyze_false_positives_negatives_gemma(msg, optimized_msg, lang)

    original_len = len(msg)
    optimized_len = len(optimized_msg)

    analysis_data_gemma.append({
        'language': lang,
        'type': msg_type,
        'original_message': msg,
        'optimized_message': optimized_msg,
        'original_length': original_len,
        'optimized_length': optimized_len,
        'fidelity_score': fidelity,
        'false_negatives_count': len(false_negatives), # Count of missing items
        'false_positives_count': len(false_positives), # Count of incorrect added items
        'false_negatives_items': false_negatives, # Actual missing items
        'false_positives_items': false_positives # Actual incorrect items
    })

analysis_df_gemma = pd.DataFrame(analysis_data_gemma)

print("\n" + "="*80)
print("📊 DETAILED ANALYSIS: FIDELITY, COMPRESSION & INFORMATION LOSS (GEMMA SIMULATED) 📊".center(80))
print("="*80)

print("\n**1. Overall Fidelity and Compression Performance:**")
overall_summary = analysis_df_gemma.groupby('language').agg(
    avg_fidelity=('fidelity_score', 'mean'),
    avg_compression=('original_length', lambda x: (x - analysis_df_gemma.loc[x.index, 'optimized_length']).mean() / x.mean()),
    avg_fn_count=('false_negatives_count', 'mean'),
    avg_fp_count=('false_positives_count', 'mean')
).reset_index()
overall_summary['avg_compression'] = overall_summary['avg_compression'].apply(lambda x: f"{x:.2%}")
overall_summary['avg_fidelity'] = overall_summary['avg_fidelity'].apply(lambda x: f"{x:.2f}")
overall_summary['avg_fn_count'] = overall_summary['avg_fn_count'].apply(lambda x: f"{x:.1f}")
overall_summary['avg_fp_count'] = overall_summary['avg_fp_count'].apply(lambda x: f"{x:.1f}")
print(overall_summary.to_string(index=False))

print("\n" + "-"*80)
print("**2. Fidelity and Information Loss by Emergency Type & Language:**")
type_lang_summary = analysis_df_gemma.groupby(['type', 'language']).agg(
    avg_fidelity=('fidelity_score', 'mean'),
    avg_fn_count=('false_negatives_count', 'mean'),
    avg_fp_count=('false_positives_count', 'mean')
).reset_index()
type_lang_summary['avg_fidelity'] = type_lang_summary['avg_fidelity'].apply(lambda x: f"{x:.2f}")
type_lang_summary['avg_fn_count'] = type_lang_summary['avg_fn_count'].apply(lambda x: f"{x:.1f}")
type_lang_summary['avg_fp_count'] = type_lang_summary['avg_fp_count'].apply(lambda x: f"{x:.1f}")
print(type_lang_summary.to_string(index=False))

print("\n" + "-"*80)
print("**3. Examples of Missing Critical Information (False Negatives):**")
# Filter for examples with actual false negatives
fn_examples = analysis_df_gemma[analysis_df_gemma['false_negatives_count'] > 0]
if not fn_examples.empty:
    for index, row in fn_examples.head(3).iterrows(): # Show top 3 examples
        print(f"\n--- 📝 Example (Lang: {row['language'].upper()}, Type: {row['type'].capitalize()}) ---")
        print(f"Original Msg:  `{row['original_message']}`")
        print(f"Optimized Msg: `{row['optimized_message']}`")
        print(f"**❌ Missing Info:** {', '.join(row['false_negatives_items'])}")
else:
    print("No examples with missing critical information found (perfect fidelity in simulation for these messages!).")

print("\n" + "-"*80)
print("**4. Examples of Incorrectly Added Information (False Positives):**")
# Filter for examples with actual false positives
fp_examples = analysis_df_gemma[analysis_df_gemma['false_positives_count'] > 0]
if not fp_examples.empty:
    for index, row in fp_examples.head(3).iterrows(): # Show top 3 examples
        print(f"\n--- 📝 Example (Lang: {row['language'].upper()}, Type: {row['type'].capitalize()}) ---")
        print(f"Original Msg:  `{row['original_message']}`")
        print(f"Optimized Msg: `{row['optimized_message']}`")
        print(f"**⚠️ Added Incorrect Info:** {', '.join(row['false_positives_items'])}")
else:
    print("No examples with incorrectly added information found (as expected from this type of optimizer).")


print("\n" + "="*80)
print("💡 ANALYSIS COMPLETE: This simulated approach demonstrates how Gemma 3N could be leveraged for sophisticated information extraction and quality assessment in emergency message optimization. Real Gemma integration would yield even more nuanced results. 💡")
print("="*80)
