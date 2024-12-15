import matplotlib.pyplot as plt
from PIL import Image

def display_images_in_batch(image_paths, batch_size=4):
    if len(image_paths) == 0:
        print("Empty image set. Nothing to display.")
        return
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        plt.figure(figsize=(15, 10))  # Adjust the figure size (width, height)

        for idx, image_path in enumerate(batch):
            image = Image.open(image_path)
            plt.subplot(1, len(batch), idx + 1)  # Create subplots (1 row, len(batch) columns)
            plt.imshow(image)
            plt.axis("off")  # Hide axes
            plt.title(f"Image {i + idx + 1}")  # Optional: Add titles to images

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

