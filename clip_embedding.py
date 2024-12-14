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
