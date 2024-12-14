import torch
from PIL import Image
import requests
import redis
import numpy as np

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

model_name = "openai/clip-vit-base-patch32"
#model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(
    model_name,
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()
model = model.to('xpu')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Extract image features (vector) using CLIP model
image_inputs = processor(images=image, return_tensors="pt", padding=True)
image_inputs = image_inputs.to("xpu")  # Move to appropriate device
with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)

image_vector = image_features.cpu().numpy().astype(np.float32).tobytes()  # Convert to bytes for Redis
# Store image vector in Redis with a unique key (e.g., "image:1")
r.set('image:1', image_vector)

#import pdb
#pdb.set_trace()

query = ["A photo of Apple", "Find me a phto of butterfly"]  # Input Query
text_inputs = tokenizer(query, padding=True, return_tensors="pt")
text_inputs = text_inputs.to("xpu")
with torch.no_grad():
    text_embeddings = model.get_text_features(**text_inputs)
# Convert text features to bytes (for storing in Redis)
text_vector = text_embeddings.cpu().numpy().astype(np.float32).tobytes()  # Convert to byte
# Store text vector in Redis with a unique key (e.g., "text:1")
r.set('text:1', text_vector)

# Example of retrieving and checking the stored vectors from Redis
stored_image_vector = np.frombuffer(r.get('image:1'), dtype=np.float32)
stored_text_vector = np.frombuffer(r.get('text:1'), dtype=np.float32)

print("Stored image vector:", stored_image_vector)
print("Stored text vector:", stored_text_vector)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
inputs = inputs.to('xpu')

with torch.no_grad():
    with torch.autocast('xpu'):
        outputs = model(**inputs)

'''
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print(logits_per_image)
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)
'''
image_inputs = processor(images=image, return_tensors="pt", padding=True)
image_inputs=image_inputs.to('xpu') #torch.bfloat16) 
with torch.no_grad():
	image_features = model.get_image_features(**image_inputs)
print(image_features)

query = ["A photo of Apple", "Find me a phto of butterfly"]  # Input Query
text_inputs = tokenizer(query, padding=True, return_tensors="pt")
text_inputs = text_inputs.to("xpu")
with torch.no_grad():
  text_embeddings = model.get_text_features(**text_inputs) 
print(text_embeddings)
