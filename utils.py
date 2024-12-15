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
