# save_model_locally.py
import os
from sentence_transformers import SentenceTransformer

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download and save model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
model.save('models/all-MiniLM-L6-v2')

print("Model saved successfully to models directory!")