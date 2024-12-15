from PIL import Image
import requests
import redis
import numpy as np
import torch
import os
from tqdm import tqdm
import time

from redis_rw import create_index, is_index_existed
from redis.commands.search.query import Query  # Import Query

from clip_embedding import CLIP_Embedding
from utils import display_images_in_batch

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process embedding input.")
parser.add_argument(
    "--embedding",
    action="store_true",
    default=False,   # Default value set to False
    help="Set embedding to True (default: False)"
)
# Parse the arguments
args = parser.parse_args()

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db = 0) #decode_responses=True)

if (not is_index_existed("myIndex")):
    create_index()

model_name = "openai/clip-vit-base-patch32"
device = "xpu"
embedding_model = CLIP_Embedding(model_name, device)

# Function to preprocess images
def preprocess_images(image_paths):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    full_image_paths = [image_path for image_path in image_paths ]
    return images, full_image_paths

# Function to embed and store images in Redis
def embed_and_store_images(image_paths, batch_size=32):
    # Step 1: Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]

        # Step 2: Preprocess the images
        images, image_path = preprocess_images(batch_paths)

        # Step 4: Get image embeddings
        image_features = embedding_model.get_image_embeddings(images) 

        # Step 5: Store embeddings in Redis
        for idx, img_path in enumerate(batch_paths):
            # Convert image features to bytes (Redis supports binary data)
            image_vector = image_features[idx].cpu().numpy().astype(np.float32).tobytes()

            # Use a unique ID for each image (you could use the image file name or a custom ID)
            image_id = f"image:{os.path.basename(img_path)}"

            # Store the image vector in Redis
            r.hset(image_id, mapping={"vector": image_vector, "id": image_id, "path": image_path[idx]})

        print(f"Batch {i//batch_size + 1} processed and stored in Redis.")

# Function to retrieve similar images based on a query
def search_similar_images(query_image_path, top_k=5):
    # Step 1: Process the query image
    query_image = Image.open(query_image_path).convert("RGB")
    # Step 2: Get image embedding for the query image
    query_embedding = embedding_model.get_image_embeddings(query_image) 

    search_images_path = search_images_by_embedding(query_embedding, top_k)
    return search_images_path 

# Function to retrieve similar images based on a query
def search_images_by_embedding(query_embedding, top_k=5):
    query_vector = np.array(query_embedding.cpu(), dtype=np.float32).tobytes()  # Convert to bytes for Redis
    query_str = f"*=>[KNN {top_k} @vector $query_vector]"  # Top 2 nearest neighbors

    # Perform the search (Redis will return the closest matches)
    search_results = r.ft("myIndex").search(
        query=Query(query_str).return_fields("id", "path", "vector").dialect(2),
        query_params={"query_vector": query_vector},
    )

    search_images_path = [result.path for result in search_results.docs]
    return search_images_path 

# Example usage
image_folder = "/home/xwang/Downloads/image"  # Folder containing images
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(".jpg")]

# Process the images and store them in Redis
if (args.embedding):
    print("Start ingest images into Database. ")
    print(f"Process images in {image_folder} Waiting for minutes ....")
    embed_and_store_images(image_paths, batch_size=32)

# Example: Querying for a similar image
query_image_path = "/home/xwang/Downloads/image/0201_18.jpg"
#search_imgae_paths = search_similar_images(query_image_path, top_k=5)
#query_image = Image.open(query_image_path).convert("RGB")
#query_embedding = embedding_model.get_image_embeddings(query_image)
#search_imgae_paths = search_images_by_embedding(query_embedding, top_k=5)

query = ["A photo of Family"] # Input Query
print("Family photo")
text_embeddings = embedding_model.embed_query(query) 
search_imgae_paths = search_images_by_embedding(text_embeddings, top_k=4)
display_images_in_batch(search_imgae_paths)

while True:
    user_input = input("Enter something (type 'exit' to quit): ").strip()  # Strip removes extra spaces
    if user_input.lower() == "exit":  # Check if the input is 'exit', case insensitive
        print("Exiting the loop. Goodbye!")
        break
    else:
        print(f"You entered: {user_input}")
        if (len(user_input) > 0):
            text_embeddings = embedding_model.embed_query(user_input) 
            search_imgae_paths = search_images_by_embedding(text_embeddings, top_k=4)
            # Display the results
            display_images_in_batch(search_imgae_paths)
