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

from utils import display_images_in_batch, filter_match_image, generate_template, extract_date_from_jpeg
from utils import list_all_directories, get_image_id
from clip_embedding import CLIP_Embedding
from blip import LVM_model
import hashlib

import argparse

def load_clip_model():    
    model_name = "openai/clip-vit-base-patch32"
    device = "xpu"
    try:
        embedding_model = CLIP_Embedding(model_name, device)
        print("Model CLIP loaded successfully!")
        return embedding_model
    except Exception as e:
        print(f"Error loading model: {e}")

def load_lvm_model():
    lvm_device = "xpu"
    lvm_model_name = "Salesforce/blip2-flan-t5-xl"
    try:
        #lvm_model_name = "/home/xwang/.cache/huggingface/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/2839125572785ff89d2438c8bf1550a98c7fcfcd"
        multiModel = LVM_model(lvm_model_name, lvm_device)
        print("Model LVM loaded successfully!")
        return multiModel
    except Exception as e:
        print(f"Error loading {lvm_model_name} model: {e}")

def connect_redis():
    try:
        # Initialize Redis connection
        redis_db = redis.Redis(host='localhost', port=6379, db = 0) #decode_responses=True)
        Index_name = "myIndex"
        if (not is_index_existed(redis_db, Index_name)):
            create_index(redis_db, Index_name)
    except Exception as e:
        print(f"Error connect_redis: {e}")

    return redis_db

# Function to preprocess images
def preprocess_images(image_paths):
    images = []
    full_image_paths = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path).convert("RGB"))
            full_image_paths.append(image_path)
        except Exception as e:
            print(f"Error read {image_path}: {e}")
    return images, full_image_paths

# Function to embed and store images in Redis
def embed_and_store_images(embedding_model, redis_db, images_path_list, batch_size=32):
    # Step 1: Process images in batches
    for i in tqdm(range(0, len(images_path_list), batch_size), desc="Processing batches"):
        img_paths_batch = images_path_list[i:i+batch_size]

        # Step 2: Preprocess the images
        images_data, images_path = preprocess_images(img_paths_batch)
        if (len(images_data) != len(images_path)):
            # process next batch if errors in preprocess images
            continue

        # Step 4: Get image embeddings
        image_features = embedding_model.get_image_embeddings(images_data) 

        # Step 5: Store embeddings in Redis
        for idx, img_path in enumerate(images_path):
            # Convert image features to bytes (Redis supports binary data)
            image_vector = image_features[idx].cpu().numpy().astype(np.float32).tobytes()

            # Use a unique ID for each image (you could use the image file name or a custom ID)
            md5_hash = hashlib.md5(img_path.encode()).hexdigest()
            image_id = f"image:{os.path.basename(img_path)}_{md5_hash}"

            date = extract_date_from_jpeg(img_path)
            # Store the image vector in Redis
            try:
                redis_db.hset(image_id, mapping={"vector": image_vector, "id": image_id, "path": img_path, "date": date})
            except Exception as e:
                print(f"Error write {img_path}: {e}")

        print(f"Batch {i//batch_size + 1} processed and stored in Redis.")

# Function to retrieve similar images based on a query
def search_similar_images(embedding_model, db, query_image_path, top_k=5):
    search_images_path = []
    # Step 1: Process the query image
    try:
        query_image = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        print(f"Error write {img_path}: {e}")

    # Step 2: Get image embedding for the query image
    if query_image is not None:
        query_embedding = embedding_model.get_image_embeddings(query_image)

    if (query_embedding is not None):
        search_images_path = search_images_by_embedding(db, query_embedding, top_k)
    return search_images_path

def search_images_by_text(embedding_model, db, input_text, top_k=5):
    search_image_paths = []
    if (input_text != ""):
        text_embeddings = embedding_model.embed_query(input_text)
        if (text_embeddings is not None):
            search_image_paths = search_images_by_embedding(db, text_embeddings, top_k)

    return search_image_paths

# Function to retrieve similar images based on a query
def search_images_by_embedding(redis_db, query_embedding, top_k):
    query_vector = np.array(query_embedding.cpu(), dtype=np.float32).tobytes()  # Convert to bytes for Redis
    query_str = f"*=>[KNN {top_k} @vector $query_vector AS score]"  # Top k nearest neighbors

    # Perform the search (Redis will return the closest matches)
    search_results = redis_db.ft("myIndex").search(
        query=Query(query_str).
        sort_by("score", asc=True).
        #return_fields("id", "path", "vector").paging(0, top_k).dialect(2),
        return_fields("id", "path", "vector", "score").paging(0, top_k).dialect(2),
        query_params={"query_vector": query_vector},
    )

    search_images_path = [result.path for result in search_results.docs]
    score = [result.score for result in search_results.docs]
    #print(score)
    return search_images_path 

def is_image_match_text(multiModel, image_lst, text):
    query_string = generate_template(text)
    questions = [query_string] * len(image_lst)
    answers = multiModel.get_image_query_answer(image_lst, questions)
    return answers

def main():
    embedding_model = load_clip_model()
    multiModel = load_lvm_model()
    redis_db = connect_redis()

    # Example usage
    #image_folder = "E:/Photo"  # Folder containing images
    #image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(".jpg")]

    # Process the images and store them in Redis
    if (args.embedding):
        print("Start ingest images into Database. ")
        if (args.folder is not None):
            root_directory = args.folder
            all_directories = list_all_directories(root_directory)
            all_directories.append(root_directory)
            for image_folder in all_directories:
                images_path_list = []
                images_path_list = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(".jpg")]
                print(f"Process images in {image_folder} Waiting for minutes ....")
                embed_and_store_images(embedding_model, redis_db, images_path_list, batch_size=128)

    query = ["A photo of Family"] # Input Query
    print("Family photo")
    text_embeddings = embedding_model.embed_query(query) 
    search_imgae_paths = search_images_by_embedding(redis_db, text_embeddings, top_k=4)
    display_images_in_batch(search_imgae_paths)

    top_k = 60
    while True:
        user_input = input("Enter something (type 'exit' to quit): ").strip()  # Strip removes extra spaces
        if user_input.lower() == "exit":  # Check if the input is 'exit', case insensitive
            print("Exiting the loop. Goodbye!")
            break
        elif "top_k" in user_input.lower():
            parts = user_input.split()
            if len(parts) > 1:
                number_str = parts[1]
                top_k = int(number_str)
        else:
            print(f"You entered: {user_input}")
            if (len(user_input) > 0):
                search_imgae_paths = search_images_by_text(embedding_model, redis_db, user_input, top_k)
                
                print("search images:", len(search_imgae_paths))
                import time
                if len(search_imgae_paths) > 0:
                    # display_images_in_batch(search_imgae_paths)
                    # Display the results
                    start = time.time()
                    answers = is_image_match_text(multiModel, search_imgae_paths, user_input)
                    duration = time.time() - start
                    print(f"bach process time: {duration}, num {top_k} ", len(search_imgae_paths))
                    match_image = filter_match_image(search_imgae_paths, answers)

                    print(f"accuracy iamges:", len(match_image))
                    display_images_in_batch(match_image)

if __name__ == '__main__':
    # Explicitly set the start method for Windows

    # Create the parser
    parser = argparse.ArgumentParser(description="Process embedding input.")
    parser.add_argument(
        "--embedding",
        action="store_true",
        default=False,   # Default value set to False
        help="Set embedding to True (default: False)"
    )

    parser.add_argument(
        "--folder",
        type=str,
        default="image_path",   # Default value set to False
        help="Set embedding to True (default: False)"
    )
    # Parse the arguments
    args = parser.parse_args()
    main()

    # Example: Querying for a similar image
    #query_image_path = "/home/xwang/Downloads/image/0201_18.jpg"
    #search_imgae_paths = search_similar_images(query_image_path, top_k=5)
    #query_image = Image.open(query_image_path).convert("RGB")
    #query_embedding = embedding_model.get_image_embeddings(query_image)
    #search_imgae_paths = search_images_by_embedding(query_embedding, top_k=5)