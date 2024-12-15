from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# Load the BLIP-2 processor and model
class LVM_model:
    def __init__(self, model_name, device):
        #model_name = "Salesforce/blip2-flan-t5-xl"
        self.processor = Blip2Processor.from_pretrained(model_name) 
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.model = self.model.eval()
        self.device = device
        return
   
    def get_image_query_answer(self, image_paths, questions): 
        # Load the input image
        #image_paths = ["0201_1.jpg", "demo.jpg"]  # Replace with the path to your image

        # Define your question
        #questions = ["Does the descritption of image right, answser yes or no. descritption: 'There is one dog in the image.'", 
        # Load and preprocess the images
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        # Preprocess the image and question
        inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True).to(self.device, torch.bfloat16)

        # Generate the answer
        outputs = self.model.generate(**inputs)

        # Decode the outputs
        answers = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]

        return answers

# Print the results
'''
for i, (question, answer) in enumerate(zip(questions, answers)):
    print(f"Image {i + 1}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-" * 30)
'''
