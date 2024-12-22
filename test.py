import tkinter as tk
from tkinter import filedialog, Label, Entry

from image_retrieval import load_clip_model, load_lvm_model, connect_redis
from image_retrieval import search_images_by_embedding, search_similar_images
from utils import display_images_in_batch, filter_match_image, generate_template

import threading

user_input = "a photo of Family"
input_image = "test.image"

def open_file_dialog():
    global input_image
    global multiModel
    global embedding_model
    global redis_db

    global entry, label_text, button_text, button_file, label_file
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    
    if file_path:
        # Update the label with the selected file path
        top_k = 30
        input_image = file_path
        label_file.config(text=f"Selected: {file_path}")
        search_imgae_paths = search_similar_images(embedding_model, redis_db, input_image, top_k)
        display_images_in_batch(search_imgae_paths)
    else:
        label_file.config(text="No file selected.")

def get_text_input():
    global user_input
    global multiModel
    global embedding_model
    global redis_db
    global entry, label_text, button_text, button_file

    # Get the input text from the entry box
    user_input = entry.get()
    label_text.config(text=f"Input: {user_input}")
    text_embeddings = embedding_model.embed_query(user_input) 
    top_k = 60
    search_imgae_paths = search_images_by_embedding(redis_db, text_embeddings, top_k)

    query_string = generate_template(user_input)
    questions = [query_string] * len(search_imgae_paths) 
    answers = multiModel.get_image_query_answer(search_imgae_paths, questions)
    match_image = filter_match_image(search_imgae_paths, answers)
    print(f"accuracy iamges:", len(match_image))
    display_images_in_batch(match_image)

def load_models():
    global embedding_model, multiModel, redis_db
    embedding_model = load_clip_model()  # Load huggingface transformers models
    multiModel = load_lvm_model()        # Load huggingface transformers models
    redis_db = connect_redis()           # Connect to Redis
    print("Models loaded successfully")

entry = None
label_text = None
button_text = None
button_file = None
label_file = None

def create_gui():
    global entry, label_text, button_text, button_file, label_file

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Image Selector with Text Input")
    root.geometry("1200x768+100+200")

    # Add an entry box for text input
    entry = Entry(root, width=40, font=("Arial", 14))
    entry.pack(pady=10)

    # Add a button to submit the text input
    button_text = tk.Button(root, text="Submit Text", font=("Arial", 14), command=get_text_input)
    button_text.pack(pady=5)

    # Add a label to display the text input
    label_text = Label(root, text="Enter text above and click submit.", font=("Arial", 14))
    label_text.pack(pady=10)

    # Add a button to open the file dialog
    button_file = tk.Button(root, text="Select Image", command=open_file_dialog, font=("Arial", 14))
    button_file.pack(pady=10)

    # Add a label to display the selected file path
    label_file = Label(root, text="No file selected.", font=("Arial", 14))
    label_file.pack(pady=10)

    # Start the Tkinter event loop
    root.mainloop()

def main():
    # Create and start a separate thread to load models
    load_models()
    create_gui()

if __name__ == '__main__':
    main()