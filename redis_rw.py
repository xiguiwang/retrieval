import redis
from redis.commands.search.field import VectorField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Check if index existed in DB
def is_index_existed(index_name):
    try:
        # Check if the index exists
        if r.ft(index_name).info():  # FT.INFO command checks if the index exists
            print("Find Index 'myIndex'...")
            return True
    except Exception as e:
        # If FT.INFO fails, the index likely doesn't exist
        print(f"Error creating index: {e}")
        print("Index 'myIndex' does not exist. Creating a new one...")
        return False

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
        print("Index "myIndex" created successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")

# Run the index creation function
#create_index()
