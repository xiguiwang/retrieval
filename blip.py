from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# Load the BLIP-2 processor and model
model_name = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("xpu")

# Load the input image
image_paths = ["0201_1.jpg", "demo.jpg"]  # Replace with the path to your image

# Define your question
#questions = ["What is in the image?", "how many dogs are in the picture?"]
questions = ["Does the descritption of image right, answser yes or no. descritption: 'There is one dog in the image.'", 
"Does the descritption of image right, answser yes or no. descritption: 'There is one dog in the image.'"] 

# Load and preprocess the images
images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

# Preprocess the image and question
inputs = processor(images=images, text=questions, return_tensors="pt", padding=True).to("xpu")

# Generate the answer
outputs = model.generate(**inputs)

# Decode the outputs
answers = [processor.decode(output, skip_special_tokens=True) for output in outputs]

# Print the results
for i, (question, answer) in enumerate(zip(questions, answers)):
    print(f"Image {i + 1}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-" * 30)
