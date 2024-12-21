import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
import os

def display_images_in_batch(image_paths, batch_size=15):
    if len(image_paths) == 0:
        print("Empty image set. Nothing to display.")
        return

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        plt.figure(figsize=(15, 10))  # Adjust the figure size (width, height)

        rows, cols = 3, 5
        for idx, image_path in enumerate(batch):
            image = Image.open(image_path)
            plt.subplot(rows, cols, idx + 1)  # Create subplots (rows, columns)
            plt.imshow(image)
            plt.axis("off")  # Hide axes
            plt.title(f"{i + idx + 1} {os.path.basename(image_path)}")  # titles to images
        plt.tight_layout()
        plt.show()

def filter_match_image(search_imgae_paths, answers):
    match_images = []
    for idx, answer in enumerate(answers):
        if (answer.lower() == 'yes'): 
            match_images.append(search_imgae_paths[idx]) 
    return match_images 

# image_paths = ["0201_1.jpg", "demo.jpg"]  # Replace with the path to your image
# Define your question
def generate_template(user_query):
    questions = f"Does the descritption match the image , answser yes or no. descritption: '{user_query}'"
        #questions = ["Does the descritption of image right, answser yes or no. descritption: 'There is one dog in the image.'", 
    return questions 


def extract_date_from_jpeg(image):
    try:
        # Open the image
        #image = Image.open(image_path)

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
