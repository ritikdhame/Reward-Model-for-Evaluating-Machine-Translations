# -*- coding: utf-8 -*-
"""Reward models in large language models (LLMs)

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QWeSy9H32MLk6pPyA535pIxJh3HpYWdn

**Step 1: Getting Dependencies**
"""

!python -m venv reward_env
!source reward_env/bin/activate  # On Windows: reward_env\Scripts\activate
!pip install torch transformers datasets pandas numpy matplotlib seaborn

"""**Step 2: Prepare the Dataset**

We’ll use a small, publicly available dataset of movie subtitles. For this example, you can use a dataset like the OpenSubtitles dataset (available via the datasets library) or manually curate a small set of English-Spanish subtitle pairs. Here’s how to load and preprocess:
"""

import pandas as pd

# Assuming a small sample dataset is uploaded to Colab (e.g., via Files upload)
# Upload a file named 'sample_en_es.csv' with columns 'en' and 'es' to Colab's file system
df = pd.read_csv("/content/sample_en_es.csv")

# Filter out any empty rows and limit to 1000 samples
df = df[(df["en"] != "") & (df["es"] != "")].head(1000)

# Store the dataset as a CSV file in Colab
df.to_csv("/content/subtitles_en_es.csv", index=False)

# Verify the data
print(f"Dataset saved with {len(df)} samples. First few rows:")
print(df.head())

"""**Step 3: Generate Translation Candidates**
We'll use the MarianMT model to generate multiple Spanish translation candidates for each English sentence in the dataset.
"""

import pandas as pd
from transformers import MarianTokenizer, MarianMTModel

# Load the dataset
df = pd.read_csv("/content/subtitles_en_es.csv")

# Load the MarianMT model and tokenizer for English-to-Spanish
model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to generate multiple translations by tweaking the decoding
def generate_translations(text, num_candidates=3):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translations = []
    for i in range(num_candidates):
        # Use different beam search parameters to get varied outputs
        outputs = model.generate(
            **inputs,
            num_beams=5 + i,  # Vary beam width
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translated_text)
    return translations

# Generate translation candidates for the dataset
df["candidates"] = df["en"].apply(lambda x: generate_translations(x))
df.to_csv("/content/subtitles_with_candidates.csv", index=False)

# Verify the data
print("Translation candidates generated. First few rows:")
print(df.head())

"""**Step 4: Simulate Human Preferences**
We'll simulate human preferences by comparing the generated translations to the reference translations using BLEU scores as a proxy for quality.
"""

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

# Load the dataset with candidates
df = pd.read_csv("/content/subtitles_with_candidates.csv")

# Compute BLEU scores for each candidate against the reference
def compute_bleu_scores(row):
    reference = row["es"].split()
    scores = []
    for candidate in eval(row["candidates"]):  # Convert string representation of list back to list
        candidate_tokens = candidate.split()
        score = sentence_bleu([reference], candidate_tokens, weights=(0.5, 0.5, 0, 0))  # Bigram BLEU
        scores.append(score)
    return scores

df["bleu_scores"] = df.apply(compute_bleu_scores, axis=1)

# Simulate preferences: the candidate with the highest BLEU score is "preferred"
df["preferred_idx"] = df["bleu_scores"].apply(lambda x: x.index(max(x)))
df.to_csv("/content/subtitles_with_preferences.csv", index=False)

# Verify the data
print("Preferences simulated. First few rows:")
print(df.head())

"""**Step 5: Train the Reward Model**
We'll train a BERT-based reward model to predict a scalar reward score for each translation candidate, learning to assign higher scores to the "preferred" translations.
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the dataset with preferences
df = pd.read_csv("/content/subtitles_with_preferences.csv")

# Load BERT tokenizer and model
reward_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
reward_model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=1)

# Prepare the dataset for training the reward model
class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df) * 3  # Each row has 3 candidates

    def __getitem__(self, idx):
        row_idx = idx // 3
        candidate_idx = idx % 3
        row = self.df.iloc[row_idx]
        candidate = eval(row["candidates"])[candidate_idx]  # Convert string representation back to list
        label = 1.0 if candidate_idx == row["preferred_idx"] else 0.0  # Binary reward

        encoding = reward_tokenizer(
            candidate,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.float)
        return encoding

# Split into train and test
train_df = df.iloc[:8]  # Small dataset (10 rows), so 80% train
test_df = df.iloc[8:]
train_dataset = RewardDataset(train_df)
test_dataset = RewardDataset(test_df)

# Training arguments (updated parameter name)
training_args = TrainingArguments(
    output_dir="/content/reward_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Smaller batch size for small dataset
    per_device_eval_batch_size=4,
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",
    logging_dir="/content/logs",
    logging_steps=10,
)

# Initialize trainer
trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the reward model
trainer.train()

# Save the model
reward_model.save_pretrained("/content/reward_model_final")
reward_tokenizer.save_pretrained("/content/reward_model_final")

"""**Step 6: Evaluate the Reward Model**
This code will load the trained reward model, score the translation candidates, calculate the accuracy of ranking preferred translations, and create a visualization of the reward score distribution.
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification

# Load the dataset with preferences
df = pd.read_csv("/content/subtitles_with_preferences.csv")

# Load the trained reward model and tokenizer
reward_tokenizer = BertTokenizer.from_pretrained("/content/reward_model_final")
reward_model = BertForSequenceClassification.from_pretrained("/content/reward_model_final")

# Function to score translations using the reward model
def score_translations(candidates):
    scores = []
    for candidate in eval(candidates):  # Convert string representation back to list
        inputs = reward_tokenizer(candidate, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = reward_model(**inputs).logits
        scores.append(outputs.item())
    return scores

# Score the test set
test_df = df.iloc[8:]  # Use the same test split as before (last 2 rows of 10)
test_df["reward_scores"] = test_df["candidates"].apply(score_translations)

# Check if the highest reward score matches the preferred translation
test_df["predicted_best_idx"] = test_df["reward_scores"].apply(lambda x: x.index(max(x)))
accuracy = (test_df["predicted_best_idx"] == test_df["preferred_idx"]).mean()
print(f"Reward Model Accuracy: {accuracy:.2f}")

# Visualize the results
plt.figure(figsize=(10, 6))
sns.boxplot(data=test_df["reward_scores"].explode())
plt.title("Distribution of Reward Scores for Translation Candidates")
plt.savefig("/content/reward_scores_distribution.png")
plt.show()