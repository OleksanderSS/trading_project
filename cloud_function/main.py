
# main.py - для Google Cloud Function

import pandas as pd
import functions_framework
import io
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google.cloud import storage

# --- Global variables for model and client ---
_FINBERT_PIPELINE = None
_STORAGE_CLIENT = None

def get_finbert_pipeline():
    """Initializes and returns a singleton FinBERT pipeline."""
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        print("Initializing FinBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        # Use device=-1 for CPU to ensure compatibility in all Cloud Function environments
        _FINBERT_PIPELINE = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
        print("FinBERT model initialized successfully.")
    return _FINBERT_PIPELINE

def get_storage_client():
    """Initializes and returns a singleton GCS client."""
    global _STORAGE_CLIENT
    if _STORAGE_CLIENT is None:
        print("Initializing GCS client...")
        _STORAGE_CLIENT = storage.Client()
        print("GCS client initialized successfully.")
    return _STORAGE_CLIENT

@functions_framework.cloud_event
def process_news_file(cloud_event):
    """Cloud function triggered by a file upload to GCS."""
    try:
        data = cloud_event.data
        bucket_name = data["bucket"]
        file_name = data["name"]

        print(f"File received: {file_name} from bucket: {bucket_name}")

        if not file_name.startswith('data/raw/'):
            print(f"File {file_name} is not in 'data/raw/', skipping.")
            return

        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # 1. Download and read the Parquet file
        print(f"Downloading {file_name}...")
        in_memory_file = io.BytesIO()
        blob.download_to_file(in_memory_file)
        in_memory_file.seek(0)
        df = pd.read_parquet(in_memory_file)
        print(f"Successfully read {len(df)} rows from {file_name}.")

        # 2. Analyze sentiment
        finbert_pipeline = get_finbert_pipeline()
        texts = df['content'].fillna('').tolist()
        
        if not texts:
            print("No content to analyze.")
            return

        print(f"Analyzing sentiment for {len(texts)} articles...")
        sentiments = finbert_pipeline(texts, truncation=True, max_length=512)
        
        label_to_score = {'positive': 1, 'negative': -1, 'neutral': 0}
        scores = [s['score'] * label_to_score.get(s['label'], 0) for s in sentiments]
        df['sentiment'] = scores
        print("Sentiment analysis complete.")

        # 3. Save processed data back to GCS
        base_filename = os.path.basename(file_name)
        output_filename = f"data/processed/{base_filename}"
        output_blob = bucket.blob(output_filename)

        print(f"Saving processed file to: {output_filename}...")
        buffer_out = io.BytesIO()
        df.to_parquet(buffer_out, index=False)
        buffer_out.seek(0)
        output_blob.upload_from_file(buffer_out, content_type="application/octet-stream")
        print(f"Successfully saved processed file to {output_filename}.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Depending on the desired retry behavior, you might want to re-raise the exception
        # raise e
