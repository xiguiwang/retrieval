import tkinter as tk
from tkinter import filedialog, Label, Entry, ttk

from image_retrieval import load_clip_model, load_lvm_model, connect_redis
from image_retrieval import search_similar_images, search_images_by_text, is_image_match_text
from utils import display_images_in_batch, filter_match_image, generate_template
from utils import resize_image_keeping_ratio
from PIL import Image, ImageTk

import webbrowser
import os
import threading
import queue
import time

from enum import Enum

class SearchType(Enum):
    BY_TEXT = 1
    BY_IMAGE = 2
    BY_TEXT_AND_IMAGE = 3

# 全局队列，用于线程间通信
thumbnail_queue = queue.Queue()

text_entry = None
button_search_text = None
button_ref_img = None
image_label = None #display selected image
button_search_image = None
scale = None
accuracy_search_checked = None
search_by_text_and_image = None
image_frame = None
canvas = None
input_image = None
display_label = None

def open_original_image(image_path):
    webbrowser.open(image_path)

def get_input_text(clear_text):
    global text_entry
    # Get the input text from the text box
    text = text_entry.get()
    if clear_text:
        text_entry.delete(0, tk.END)
    return text

def clear_display_image():
    global image_frame, canvas

    for widget in image_frame.winfo_children():
        widget.destroy()
    canvas.configure(scrollregion=canvas.bbox("all"))

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

def is_accurate_search():
    global accuracy_search_checked
    if accuracy_search_checked.get():
        accuracy_search = True
    else:
        accuracy_search = False
    return accuracy_search

def is_search_by_txt_and_img():
    global search_by_text_and_image
    if search_by_text_and_image.get():
        search_by_txt_and_img = True
    else:
        search_by_txt_and_img = False
    return search_by_txt_and_img

def search_by_image():
    global input_image
    global multiModel
    global embedding_model
    global redis_db
    global scale

    accuracy_search = is_accurate_search()
    top_k = scale.get()
    search_type = SearchType.BY_IMAGE

    srch_by_txt_and_img = is_search_by_txt_and_img()

    if (input_image):
        if accuracy_search:
            input_text = get_input_text(clear_text = True)
            if srch_by_txt_and_img and input_text:
                text = input_text
            else:
                question = "Please describle the picture."
                input_image_lst =  [input_image]
                question_lst = [question]
                output = multiModel.get_image_query_answer(input_image_lst, question_lst)
                text = output[0]
                print("image:", text)
        else:
            text = None
        button_search_text.config(state=tk.DISABLED)
        button_search_image.config(state=tk.DISABLED)
        #threading.Thread(target=search_images_and_display, args = (top_k,),
        threading.Thread(target=search_images_and_display,
                         args = (embedding_model, multiModel,
                                 redis_db, input_image, text,
                                 search_type, accuracy_search, top_k),
                         daemon=True).start()
    #display_images_in_batch(search_imgae_paths)

def search_by_text():
    global multiModel
    global embedding_model
    global redis_db
    global text_entry, button_search_text
    global scale

    accuracy_search = is_accurate_search()
    top_k = scale.get()
    input_text = get_input_text(clear_text = False)
    search_type = SearchType.BY_TEXT

    if (input_text):
        button_search_text.config(state=tk.DISABLED)
        button_search_image.config(state=tk.DISABLED)
        #threading.Thread(target=search_images_and_display, args = (top_k,),
        threading.Thread(target=search_images_and_display,
                         args = (embedding_model, multiModel,
                                 redis_db, None, input_text,
                                 search_type, accuracy_search, top_k),
                         daemon=True).start()

def search_images_and_display(Embed_model, Lvm_model, db,
                              ref_image, text,
                              search_type, accuracy_search = False, top_k = 5):
    global thumbnail_queue
    clear_display_image()
    if search_type == SearchType.BY_IMAGE:
        start = time.time()
        search_images_path = search_similar_images(embedding_model, db, ref_image, top_k)
        duration = time.time() - start
    elif search_type == SearchType.BY_TEXT:
        start = time.time()
        search_images_path = search_images_by_text(embedding_model, db, text, top_k)
        duration = time.time() - start
    elif search_type == SearchType.BY_TEXT_AND_IMAGE:
        print("searet type", search_type)
    else:
        print("Unsupported searet type", search_type)

    info = f"Search Number: {top_k}, accurate search: {accuracy_search}, \
          Search by: {search_type.name}, {text}, time: {duration}"
    display_label.config(text=info)

    thumbnails = []
    batch_size = 32  # 每次处理的图片数量
    for i in range(0, len(search_images_path), batch_size):
        batch_images = search_images_path[i:i + batch_size]
        if (accuracy_search and text):
            answers = is_image_match_text(multiModel, batch_images, text)
            display_images = filter_match_image(batch_images, answers)
        else:
            display_images = batch_images

        for image_path in display_images:
            image = Image.open(image_path)
            image.thumbnail((200,200))
            photo = ImageTk.PhotoImage(image)
            thumbnails.append((photo, image_path))
        # 将 thumbnails 放入队列
        print(f"generate {i}/{len(display_images)} ")
        thumbnail_queue.put(thumbnails)
        thumbnails = []

        # 通知主线程更新 UI
        root.after(0, process_thumbnail_queue)
    button_search_text.config(state=tk.NORMAL)
    button_search_image.config(state=tk.NORMAL)

def process_thumbnail_queue():
    global thumbnail_queue

    """从队列中取出 thumbnails 并更新 UI"""
    while not thumbnail_queue.empty():
        thumbnails = thumbnail_queue.get()
        update_ui(thumbnails)


def update_ui(thumbnails):
    global image_frame
    global canvas

    columns = 5
    row, col = len(image_frame.winfo_children()) // columns, len(image_frame.winfo_children()) % columns

    print("display batch_images", len(thumbnails))
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


def load_models():
    global embedding_model, multiModel, redis_db
    embedding_model = load_clip_model()  # Load huggingface transformers models
    multiModel = load_lvm_model()        # Load huggingface transformers models
    redis_db = connect_redis()           # Connect to Redis
    print("Models loaded successfully")

def create_gui():
    global text_entry, button_search_text, button_ref_img, root, image_label
    global scale, accuracy_search_checked, button_search_image
    global image_frame, canvas, display_label, search_by_text_and_image

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Image Management: Image Search by Text and Image")
    root.geometry("1200x768+100+200")

    scale_frame = tk.Frame(root)
    scale_frame.pack(pady=5)

    scale = tk.Scale(scale_frame, from_=1, to=200, length = 400, orient=tk.HORIZONTAL)
    scale.pack(side=tk.LEFT, padx = 10, pady=10)

    accuracy_search_checked = tk.BooleanVar()
    # Create check boxes
    checkbutton = tk.Checkbutton(scale_frame, text="Accurate Search", variable=accuracy_search_checked)
    checkbutton.pack(side=tk.LEFT, padx = 10, pady=10)

    search_by_text_and_image = tk.BooleanVar()
    # Create check boxes
    checkbutton2 = tk.Checkbutton(scale_frame, text="search_by_text_and_image", variable=search_by_text_and_image)
    checkbutton2.pack(side=tk.LEFT, padx = 10, pady=10)

    text_frame = tk.Frame(root)
    text_frame.pack(pady=5)

    # Add an text box for text input
    text_entry = Entry(text_frame, width=40, font=("Arial", 14))
    text_entry.pack(side=tk.LEFT, padx = 10, pady=10)


    # Add a button to submit search_by_text
    button_search_text = tk.Button(text_frame, text="Search Image by Text", font=("Arial", 14), command=search_by_text)
    button_search_text.pack(side=tk.LEFT, padx = 10, pady=10)

    # Add a label to display the selected image
    image_label = tk.Label(root)
    image_label.pack(side=tk.LEFT, anchor="nw", padx = 10, pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    display_label = tk.Label(root, text="")
    display_label.pack(pady=10)

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