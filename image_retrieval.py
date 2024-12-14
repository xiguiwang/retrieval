import torch
from PIL import Image
import requests
import redis
import numpy as np

#from readis_rw import create_index

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

model_name = "openai/clip-vit-base-patch32"
device = "xpu"

class CLIP_Embedding:
    def __init__(self, model_name, device):
        #super().__init__()
        #model_name = "openai/clip-vit-base-patch16"
        self.model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
        ).eval()
        self.device = device
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        return

    def embed_query(self, texts):
        """Input is list of texts."""
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        text_inputs = text_inputs.to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        return text_features

    def get_embedding_length(self):
        text_features = self.embed_query("sample_text")
        return text_features.shape[1]

    def get_image_embeddings(self, images):
        """Input is list of images."""
        image_inputs = self.processor(images=images, padding=True, return_tensors="pt")
        image_inputs = image_inputs.to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
        return image_features


embedding_model = CLIP_Embedding(model_name, device)

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
fimage1 = "000000039769.jpg"
fimage2 = "0201_1.jpg"
image1 = Image.open(fimage1)
image2 = Image.open(fimage2)
images = [image1, image2]

import pdb
pdb.set_trace()
image_features = embedding_model.get_image_embeddings(images) 

image_vector = image_features.cpu().numpy().astype(np.float32).tobytes()  # Convert to bytes for Redis
# Store image vector in Redis with a unique key (e.g., "image:1")
image_key = "image:1"
r.hset(image_key, mapping={"vector": image_vector, "id": "image_1"})

# Extract text features (vector) using CLIP model
query = ["A photo of Apple", "Find me a phto of butterfly"]  # Input Query
text_vector = embedding_model.embed_query(query) 
# Convert text features to bytes (for storing in Redis)
text_vector = text_embeddings.cpu().numpy().astype(np.float32).tobytes()  # Convert to byte
# Store text vector in Redis (use a unique key)
text_key = "text:1"
r.hset(text_key, mapping={"vector": text_vector, "id": "text_1"})

# Example of retrieving and checking the stored vectors from Redis
stored_image_vector = np.frombuffer(r.get('image:1'), dtype=np.float32)
stored_text_vector = np.frombuffer(r.get('text:1'), dtype=np.float32)

print("Stored image vector:", stored_image_vector)
print("Stored text vector:", stored_text_vector)

# Extract image and text features (vector) using CLIP model
'''
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image1, return_tensors="pt", padding=True)
inputs = inputs.to('xpu')

with torch.no_grad():
    with torch.autocast('xpu'):
        outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print(logits_per_image)
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)
'''
