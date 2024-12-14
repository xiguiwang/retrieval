from PIL import Image
import requests
import redis
import numpy as np
import torch
import os
from tqdm import tqdm
import time

from redis_rw import create_index
from clip_embedding import CLIP_Embedding
from redis.commands.search.query import Query  # Import Query

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db = 0) #decode_responses=True)

create_index()

model_name = "openai/clip-vit-base-patch32"
device = "xpu"
embedding_model = CLIP_Embedding(model_name, device)

# Function to preprocess images
def preprocess_images(image_paths):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    return images

# Function to embed and store images in Redis
def embed_and_store_images(image_paths, batch_size=32):
    # Step 1: Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]

        # Step 2: Preprocess the images
        images = preprocess_images(batch_paths)

        # Step 4: Get image embeddings
        image_features = embedding_model.get_image_embeddings(images) 

        # Step 5: Store embeddings in Redis
        for idx, img_path in enumerate(batch_paths):
            # Convert image features to bytes (Redis supports binary data)
            image_vector = image_features[idx].cpu().numpy().astype(np.float32).tobytes()

            # Use a unique ID for each image (you could use the image file name or a custom ID)
            image_id = f"image:{os.path.basename(img_path)}"

            # Store the image vector in Redis
            r.hset(image_id, mapping={"vector": image_vector, "id": image_id})

        print(f"Batch {i//batch_size + 1} processed and stored in Redis.")

# Function to retrieve similar images based on a query
def search_similar_images(query_image_path, top_k=5):
    # Step 1: Process the query image
    query_image = Image.open(query_image_path).convert("RGB")

    # Step 2: Get image embedding for the query image
    query_embedding = embedding_model.get_image_embeddings(query_image) 
    query_vector = np.array(query_embedding.cpu(), dtype=np.float32).tobytes()  # Convert to bytes for Redis
    
    query_str = "*=>[KNN 5 @vector $query_vector]"  # Top 2 nearest neighbors
    # Perform the search (Redis will return the closest matches)
    search_results = r.ft("myIndex").search(
        query=Query(query_str).return_fields("id", "vector", "vecotr_score").dialect(2),
        query_params={"query_vector": query_vector},
    )

    #search_results = r.ft("myIndex").search(query_vector, query_params={"K": top_k})
    pre_path = os.path.dirname(query_image_path)    

    # Print the results
    for result in search_results.docs:
        #image.show
        file_name = result.id.split(":")[1]
        print(file_name)
        full_path = os.path.join(pre_path, file_name)
        image = Image.open(full_path)
        image.show(full_path)
        #time.sleep(5)
        #print(f"Found similar image: {result.id} with score: {result.vector_score}")

# Example usage
image_folder = "/home/xwang/Downloads/image"  # Folder containing images
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(".jpg")]

# Process the images and store them in Redis
embed_and_store_images(image_paths, batch_size=32)

# Example: Querying for a similar image
query_image_path = "/home/xwang/Downloads/image/0201_18.jpg"
search_similar_images(query_image_path, top_k=5)

import pdb
pdb.set_trace()
exit(1)

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

query_str = "*=>[KNN 5 @vector $query_vector]"  # Top 2 nearest neighbors
# Perform the vector search using RediSearch
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
