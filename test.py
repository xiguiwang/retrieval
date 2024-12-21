import tkinter as tk
from tkinter import filedialog, Label, Entry

def open_file_dialog():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    
    if file_path:
        # Update the label with the selected file path
        label_file.config(text=f"Selected: {file_path}")
    else:
        label_file.config(text="No file selected.")

def get_text_input():
    # Get the input text from the entry box
    user_input = entry.get()
    label_text.config(text=f"Input: {user_input}")

# Create the main Tkinter window
root = tk.Tk()
root.title("Image Selector with Text Input")
root.geometry("500x300")

# Add an entry box for text input
entry = Entry(root, width=40)
entry.pack(pady=10)

# Add a button to submit the text input
button_text = tk.Button(root, text="Submit Text", command=get_text_input)
button_text.pack(pady=5)

# Add a label to display the text input
label_text = Label(root, text="Enter text above and click submit.")
label_text.pack(pady=10)

# Add a button to open the file dialog
button_file = tk.Button(root, text="Select Image", command=open_file_dialog)
button_file.pack(pady=10)

# Add a label to display the selected file path
label_file = Label(root, text="No file selected.")
label_file.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
