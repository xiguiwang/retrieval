import tkinter as tk
from tkinter import filedialog, Label, Entry, ttk

from image_retrieval import load_clip_model, load_lvm_model, connect_redis
from image_retrieval import search_images_by_embedding, search_similar_images
from utils import display_images_in_batch, filter_match_image, generate_template
from utils import resize_image_keeping_ratio
from PIL import Image, ImageTk

import webbrowser
import os
import threading

user_input = "a photo of Family"
input_image = "test.image"

entry = None
button_search_text = None
button_ref_img = None
image_label = None #display selected image
button_search_image = None
scale = None
accuracy_search_checked = None
image_frame = None
canvas = None

def open_ref_image():
    global input_image

    file_path = filedialog.askopenfilename(
        title = "Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        input_image = file_path
        image = Image.open(file_path)
        image.thumbnail((200,200))
        photo = ImageTk.PhotoImage(image)

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

    if (input_image):
        button_search_text.config(state=tk.DISABLED)
        button_search_image.config(state=tk.DISABLED)
        threading.Thread(target=search_images_and_display, args = (embedding_model, redis_db, input_image, top_k,),
        #threading.Thread(target=search_images_and_display, args = (top_k,),
            daemon=True).start()
    #display_images_in_batch(search_imgae_paths)

def search_images_and_display(embedding_model, redis_db, input_image, top_k):
#def search_images_and_display(top_k):
    global image_frame, canvas

    for widget in image_frame.winfo_children():
        widget.destroy()
    canvas.configure(scrollregion=canvas.bbox("all"))

    thumbnails = []
    search_images_path = search_similar_images(embedding_model, redis_db, input_image, top_k)
    '''
    search_images_path = filedialog.askopenfilenames(
        title = "Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    '''
    for image_path in search_images_path:
        image = Image.open(image_path)
        image.thumbnail((200,200))
        photo = ImageTk.PhotoImage(image)
        thumbnails.append((photo, image_path))

    root.after(0, update_ui, thumbnails)

def update_ui(thumbnails):
    global image_frame
    global canvas

    columns = 5
    row, col = len(image_frame.winfo_children()) // columns, len(image_frame.winfo_children()) % columns

    for photo, file_path in thumbnails:
        thumbnail_label = tk.Label(image_frame, image=photo)
        thumbnail_label.image = photo
        thumbnail_label.grid(row=row, column=col, padx = 5, pady = 5)

        thumbnail_label.bind("<Button-1>", lambda e, path=file_path: open_original_image(path))

        col += 1
        if col >= columns:
            col = 0
            row += 1

        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        button_search_text.config(state=tk.NORMAL)
        button_search_image.config(state=tk.NORMAL)

def open_original_image(image_path):
    webbrowser.open(image_path)

def search_by_text():
    global user_input
    global multiModel
    global embedding_model
    global redis_db
    global entry, button_search_text
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

def create_gui():
    global entry, button_search_text, button_ref_img, root, image_label
    global scale, accuracy_search_checked, button_search_image
    global image_frame, canvas

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Image Management: Image Search by Text and Image")
    root.geometry("1200x768+100+200")

    scale_frame = tk.Frame(root)
    scale_frame.pack(pady=5)

    scale = tk.Scale(scale_frame, from_=0, to=200, length = 400, orient=tk.HORIZONTAL)
    scale.pack(side=tk.LEFT, padx = 10, pady=10)

    accuracy_search_checked = tk.BooleanVar()
    # Create check boxes
    checkbutton = tk.Checkbutton(scale_frame, text="Accurate Search", variable=accuracy_search_checked)
    checkbutton.pack(side=tk.LEFT, padx = 10, pady=10)

    text_frame = tk.Frame(root)
    text_frame.pack(pady=5)

    # Add an entry box for text input
    entry = Entry(text_frame, width=40, font=("Arial", 14))
    entry.pack(side=tk.LEFT, padx = 10, pady=10)


    # Add a button to submit search_by_text
    button_search_text = tk.Button(text_frame, text="Search Image by Text", font=("Arial", 14), command=search_by_text)
    button_search_text.pack(side=tk.LEFT, padx = 10, pady=10)

    # Add a label to display the selected image
    image_label = tk.Label(root)
    image_label.pack(side=tk.LEFT, anchor="nw", padx = 10, pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    # Add a button to open the file dialog
    button_ref_img = tk.Button(button_frame, text="Select Reference Image", command=open_ref_image, font=("Arial", 14))
    button_ref_img.pack(side=tk.LEFT, padx = 5, pady=10)

    # Add a button to submit search_by_image
    button_search_image = tk.Button(button_frame, text="Search Image by Image", font=("Arial", 14), command=search_by_image)
    button_search_image.pack(side=tk.LEFT, padx = 5, pady=10)

    # create Canvas and scrollbar
    canvas = tk.Canvas(root)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    image_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=image_frame, anchor="nw")

    # Start the Tkinter event loop
    root.mainloop()


def main():
    # Create and start a separate thread to load models
    load_models()
    create_gui()

if __name__ == '__main__':
    main()