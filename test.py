
from PIL import Image
from PIL.ExifTags import TAGS

import redis

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

def extract_date_from_jpeg(image_path):
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Extract EXIF metadata
        exif_data = image._getexif()
        if exif_data is not None:
            # Search for the 'DateTimeOriginal' or 'DateTime' field
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTime"):
                    return value  # Return the date and time
        return "No EXIF date found in the image."
    except Exception as e:
        return f"Error: {e}"

# Example usage
image_path = "0201_1.jpg"
date = extract_date_from_jpeg(image_path)
print(f"Date extracted from image: {date}")


# Retrieve the data
#image_data = get_image_data(r, "12345")
#print("Retrieved data:", image_data)
