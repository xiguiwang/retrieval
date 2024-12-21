import redis
import numpy as np
from PIL import Image

from clip_embedding import CLIP_Embedding

model_name = "openai/clip-vit-base-patch32"
device = "xpu"
embedding_model = CLIP_Embedding(model_name, device)

# Connect to Redis
r = redis.StrictRedis(host="localhost", port=6379, db=0)

image_ids = r.keys("image:*")
# Decode and print the keys
image_key_ids = [key.decode("utf-8") for key in image_ids]

## Retrieve the stored data
#image_id = "image:your_image_name.jpg"  # Replace with the correct image ID
for key in image_key_ids:
    stored_data = r.hget(key, "vector")  # Get the "vector" field from the Redis hash
    image_path = r.hget(key, "path")  # Get the "vector" field from the Redis hash
    if stored_data:
        # Convert the binary data back to a NumPy array
        print(image_path)
        image_vector = np.frombuffer(stored_data, dtype=np.float32)

        query_image = Image.open(image_path).convert("RGB")
        query_embedding = embedding_model.get_image_embeddings(query_image) 
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
        #print("Decoded vector:", query_embedding)
        print(np.mean(np.abs(query_embedding - image_vector)))
    else:
        print(f"No vector found for ID: {image_id}")

