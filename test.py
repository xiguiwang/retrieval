import tkinter as tk
from tkinter import filedialog, Label, Entry

from image_retrieval import load_clip_model, load_lvm_model, connect_redis
from image_retrieval import search_images_by_embedding, search_similar_images
from utils import display_images_in_batch, filter_match_image, generate_template
from utils import resize_image_keeping_ratio
from PIL import Image, ImageTk

import threading

user_input = "a photo of Family"
input_image = "test.image"

def open_file_dialog():
    global input_image
    global multiModel
    global embedding_model
    global redis_db

    global entry, button_text, button_file, image_label
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    
    if file_path:
        # Update the label with the selected file path
        top_k = 30
        input_image = file_path
        image = Image.open(input_image)
        # Resize the image if needed
        target_size = (300, 300)
        resize_image = resize_image_keeping_ratio(image, target_size)
        
        photo = ImageTk.PhotoImage(resize_image)

        image_label.config(image=photo)
        image_label.image = photo
        #search_imgae_paths = search_similar_images(embedding_model, redis_db, input_image, top_k)
        #display_images_in_batch(search_imgae_paths)


def search_by_image():
    global input_image
    global multiModel
    global embedding_model
    global redis_db
    global scale, accuracy_search_checked

    if accuracy_search_checked.get():
        accuracy_search = True
    else:
        accuracy_search = False

    top_k = scale.get()
    print("Search Number:", top_k)
    search_imgae_paths = search_similar_images(embedding_model, redis_db, input_image, top_k)
    display_images_in_batch(search_imgae_paths)


def search_by_text():
    global user_input
    global multiModel
    global embedding_model
    global redis_db
    global entry, button_text, button_file
    global scale, accuracy_search_checked

    if accuracy_search_checked.get():
        accuracy_search = True
    else:
        accuracy_search = False

    top_k = scale.get()
    print("Search Number:", top_k)

    # Get the input text from the entry box
    user_input = entry.get()
    if (user_input != ""):
        text_embeddings = embedding_model.embed_query(user_input) 
    else:
        return

    search_imgae_paths = search_images_by_embedding(redis_db, text_embeddings, top_k)

    if (accuracy_search):
        query_string = generate_template(user_input)
        questions = [query_string] * len(search_imgae_paths) 
        answers = multiModel.get_image_query_answer(search_imgae_paths, questions)
        match_image = filter_match_image(search_imgae_paths, answers)
        print(f"accuracy images:", len(match_image))
        display_images_in_batch(match_image)
    else:
        print(f"Fast search images:", len(search_imgae_paths))
        display_images_in_batch(search_imgae_paths)

def load_models():
    global embedding_model, multiModel, redis_db
    embedding_model = load_clip_model()  # Load huggingface transformers models
    multiModel = load_lvm_model()        # Load huggingface transformers models
    redis_db = connect_redis()           # Connect to Redis
    print("Models loaded successfully")

entry = None
button_text = None
button_file = None
image_label = None #display selected image
button_image = None
scale = None
accuracy_search_checked = None


def create_gui():
    global entry, button_text, button_file, root, image_label
    global scale, accuracy_search_checked

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Image Selector with Text Input")
    root.geometry("1200x768+100+200")

    scale = tk.Scale(root, from_=0, to=100, length = 400, orient=tk.HORIZONTAL)
    scale.grid(row=0, column=0)
    #scale.pack()

    accuracy_search_checked = tk.BooleanVar()
    # Create check boxes
    checkbutton1 = tk.Checkbutton(root, text="Accurate Search", variable=accuracy_search_checked)
    #checkbutton1.pack()
    checkbutton1.grid(row=0, column=1)

    # Add an entry box for text input
    entry = Entry(root, width=40, font=("Arial", 14))
    #entry.pack(pady=10)
    entry.grid(row=1, column=0, columnspan=3, pady=10)

    # Add a button to submit search_by_text
    button_text = tk.Button(root, text="Search Image by Text", font=("Arial", 14), command=search_by_text)
    button_text.grid(row=1, column=2, columnspan=3, pady=5)

    # Add a label to display the selected image
    image_label = tk.Label(root)
    image_label.grid(row=2, column=1, columnspan=3)

    # Add a button to open the file dialog
    button_file = tk.Button(root, text="Select Image", command=open_file_dialog, font=("Arial", 14))
    button_file.grid(row=3, column=1, columnspan=3, pady=10)

    # Add a button to submit search_by_image
    button_image = tk.Button(root, text="Search Image by Image", font=("Arial", 14), command=search_by_image)
    button_image.grid(row=4, column=1, columnspan=3, pady=5)

    # Start the Tkinter event loop
    root.mainloop()


def main():
    # Create and start a separate thread to load models
    load_models()
    create_gui()

if __name__ == '__main__':
    main()