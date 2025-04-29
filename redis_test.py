import redis
import numpy as np
from PIL import Image

from clip_embedding import CLIP_Embedding

# Connect to Redis
r = redis.StrictRedis(host="localhost", port=6379, db=1)

def load_clip():
    model_name = "openai/clip-vit-base-patch32"
    device = "xpu"
    embedding_model = CLIP_Embedding(model_name, device)
    return embedding_model

def compare_db_vecotr_match(image_path, img_db_vector):
    image_vector = np.frombuffer(stored_data, dtype=np.float32)
    query_image = Image.open(image_path).convert("RGB")
    query_embedding = embedding_model.get_image_embeddings(query_image) 
    # Convert the binary data back to a NumPy array
    query_embedding = query_embedding.cpu().numpy().astype(np.float32)
    #print("Decoded vector:", query_embedding)
    return np.mean(np.abs(query_embedding - image_vector))

def write_db_image_path():
    image_ids = r.keys("image:*")
    # Decode and print the keys
    image_key_ids = [key.decode("utf-8") for key in image_ids]
    #import pdb
    #pdb.set_trace()
    ## Retrieve the stored data
    with open('db_image_list.txt', 'w', encoding='utf-8') as file:
        for key in image_key_ids:
            #stored_vector = r.hget(key, "vector")  # Get the "vector" field from the Redis hash
            image_path = r.hget(key, "path")  # Get the "vector" field from the Redis hash
            id_value = r.hget(key, "id").decode('utf-8')  # Get the "vector" field from the Redis hash
            id_value
            if image_path:
                #vec_diff = compare_db_vecotr_match (image_path, stored_vector)
                file.write(id_value + image_path.decode('utf-8') + '\n')
                #print(image_path.decode('utf-8'))
            else:
                print(f"No vector found for ID: {image_id}")

write_db_image_path()