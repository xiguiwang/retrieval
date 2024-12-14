import redis
import numpy as np

# Connect to Redis Stack running locally in Docker
r = redis.Redis(host='localhost', port=6379, db=0)

# Example: Add a vector to Redis Stack (ensure you're using the correct vector data type)
vector = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()

# Store the vector in Redis
r.set('vector_key', vector)

# Retrieve the vector
stored_vector = np.frombuffer(r.get('vector_key'), dtype=np.float32)

print("Stored vector:", stored_vector)

