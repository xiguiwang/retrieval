import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import os
import hashlib

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
        return "1900-01-01"

def list_all_directories(root_dir):
    root_path = Path(root_dir)
    return [str(p) for p in root_path.rglob('*') if p.is_dir()]


def test_list_dirs(root_directory):
    all_directories = list_all_directories(root_directory)

    img_sum = 0
    with open('image_list.txt', 'w', encoding='utf-8') as file:
        for image_folder in all_directories:
            print(f"Process images in {image_folder} Waiting for minutes ....")
            image_paths = []
            image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(".jpg")]
            img_count = len(image_paths)
            img_sum = img_sum + len(image_paths)
            sum_str = f"{image_folder}: Number of Images {img_count}, Total {img_sum}"
            file.write(sum_str + '\n')
            for path in image_paths:
                file.write(path + '\n')
            print(sum_str)
        print(sum_str)


def get_image_id(image_file):
    # Use a unique ID for each image (you could use the image file name or a custom ID)
    image_id_old = f"image:{os.path.basename(image_file)}"

    md5_hash = hashlib.md5(image_file.encode()).hexdigest()
    image_id_new = f"{image_id_old}_{md5_hash}"

    return image_id_new

def resize_image_keeping_ratio(image, target_size):
    width, height = image.size
    target_width, target_height = target_size

    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # The image is wider than the target aspect ratio
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # The image is taller than the target aspect ratio
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new blank image with the target size and paste the resized image onto it
    '''
    final_image = Image.new("RGB", target_size)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    final_image.paste(resized_image, (x_offset, y_offset))
    '''

    return resized_image
    