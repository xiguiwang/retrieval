import redi

# Retrieve data for a specific image ID
def get_image_data(redis_client, image_id):
    vector_key = f"image:{image_id}"
    data = redis_client.hgetall(vector_key)
    
    # Decode the vector from binary
    vector = np.frombuffer(data[b"vector"], dtype=np.float32)
    
    # Decode other fields
    image_path = data[b"path"].decode("utf-8")
    image_id = data[b"id"].decode("utf-8")
    image_date = data[b"date"].decode("utf-8")
    
    return {
        "vector": vector,
        "path": image_path,
        "id": image_id,
        "date": image_date
    }

# Retrieve the data
image_data = get_image_data(r, "12345")
print("Retrieved data:", image_data)
