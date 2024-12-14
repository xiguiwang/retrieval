from PIL import Image
import requests
import redis
import numpy as np
import torch

from redis_rw import create_index
from clip_embedding import CLIP_Embedding
from redis.commands.search.query import Query  # Import Query

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db = 0) #decode_responses=True)

create_index()

model_name = "openai/clip-vit-base-patch32"
device = "xpu"
embedding_model = CLIP_Embedding(model_name, device)

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"

import pdb
pdb.set_trace()

image_features = embedding_model.get_image_embeddings(images) 

image_vector = image_features[0].cpu().numpy().astype(np.float32).tobytes()  # Convert to bytes for Redis
# Store image vector in Redis with a unique key (e.g., "image:1")
image_key = "image:0"
r.hset(image_key, mapping={"vector": image_vector, "id": "image_1"})

image_vector = image_features[1].cpu().numpy().astype(np.float32).tobytes()  # Convert to bytes for Redis
image_key = "image:1"
r.hset(image_key, mapping={"vector": image_vector, "id": "image_1"})

image_key = "image:2"
r.hset(image_key, mapping={"vector": image_vector, "id": "image_1"})

# Extract text features (vector) using CLIP model
query = ["A photo of Apple", "Find me a phto of butterfly"]  # Input Query
text_embeddings = embedding_model.embed_query(query) 
# Convert text features to bytes (for storing in Redis)
text_vector = text_embeddings.cpu().numpy().astype(np.float32).tobytes()  # Convert to byte
# Store text vector in Redis (use a unique key)
text_key = "text:1"
r.hset(text_key, mapping={"vector": text_vector, "id": "text_1"})

# Query vector (for example, a random vector for testing)
query_vector = np.random.rand(512).astype(np.float32).tobytes()  # Create a random query vector

query_str = "*=>[KNN 2 @vector $query_vector]"  # Top 2 nearest neighbors

# Perform the vector search using RediSearch
query_str = "*=>[KNN 1 @vector $query_vector]"  # Top 2 nearest neighbors

search_results = r.ft("myIndex").search(
    query=Query(query_str).return_fields("id", "vector", "vecotr_score").dialect(2),
    query_params={"query_vector": query_vector},
)

# Print out the results (IDs and similarities)
for result in search_results.docs:
    print(f"ID: {result.id}, Similarity: {result.vector_score}")

'''
# Extract image and text features (vector) using CLIP model
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
