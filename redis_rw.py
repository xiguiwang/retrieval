import redis
from redis.commands.search.field import VectorField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Create a Redis Search index for image and text vectors
def create_index():
    try:
        # Define the vector dimension (e.g., 512 for CLIP model)
        vector_dimension = 512
        
        # Define the index for storing image and text vectors
        r.ft("myIndex").create_index([
            VectorField("vector", "FLAT", { "TYPE": "float32", "DIM": vector_dimension, "DISTANCE_METRIC": "COSINE" }),
            TagField("id")  # Store the ID for easy lookup
        ], definition=IndexDefinition(prefix=["image:"], index_type=IndexType.HASH))
        print("Index created successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")

# Run the index creation function
#create_index()
