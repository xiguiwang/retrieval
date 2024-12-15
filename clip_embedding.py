import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

class CLIP_Embedding:
    def __init__(self, model_name, device):
        #super().__init__()
        #model_name = "openai/clip-vit-base-patch16"
        self.model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
        ).eval()
        self.device = device
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        return

    def embed_query(self, texts):
        """Input is list of texts."""
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        text_inputs = text_inputs.to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        return text_features

    def get_embedding_length(self):
        text_features = self.embed_query("sample_text")
        return text_features.shape[1]

    def get_image_embeddings(self, images):
        """Input is list of images."""
        image_inputs = self.processor(images=images, padding=True, return_tensors="pt")
        image_inputs = image_inputs.to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
        return image_features

'''
# Extract text features (vector) using CLIP model
query = ["A photo of Apple", "Find me a phto of butterfly"]  # Input Query
text_embeddings = embedding_model.embed_query(query) 
# Convert text features to bytes (for storing in Redis)
text_vector = text_embeddings.cpu().numpy().astype(np.float32).tobytes()  # Convert to byte
# Store text vector in Redis (use a unique key)
text_key = "text:1"
r.hset(text_key, mapping={"vector": text_vector, "id": "text_1"})

# Query vector (for example, a random vector for testing)
query_vector = np.random.rand(512).astype(np.float32).tobytes()  # Create a random query vector

query_str = "*=>[KNN 5 @vector $query_vector]"  # Top 2 nearest neighbors
# Perform the vector search using RediSearch
search_results = r.ft("myIndex").search(
    query=Query(query_str).return_fields("id", "vector", "path").dialect(2),
    query_params={"query_vector": query_vector},
)

# Extract image and text features (vector) using CLIP model
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image1, return_tensors="pt", padding=True)
inputs = inputs.to('xpu')

with torch.no_grad():
    with torch.autocast('xpu'):
        outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print(logits_per_image)
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)
'''
