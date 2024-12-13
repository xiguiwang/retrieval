import torch
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

import pdb
#model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    #attn_implementation="flash_attention_2",
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
)
#pdb.set_trace()
#device_map='xpu',
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()
model = model.to('xpu')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
inputs = inputs.to('xpu')

with torch.no_grad():
    with torch.autocast('xpu'):
        outputs = model(**inputs)
#outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print(logits_per_image)
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)


image_inputs = processor(images=image, return_tensors="pt", padding=True)
image_inputs=image_inputs.to('xpu') #torch.bfloat16) 
with torch.no_grad():
	image_features = model.get_image_features(**image_inputs)
print(image_features)

import pdb
pdb.set_trace()

query = ["A photo of Apple", "Find me a phto of butterfly"]  # Input Query
text_inputs = tokenizer(query, padding=True, return_tensors="pt")
text_inputs = text_inputs.to("xpu")
with torch.no_grad():
  text_embeddings = model.get_text_features(**text_inputs) 
print(text_embeddings)


class vCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_frm = cfg["num_frm"]
        self.model_name = cfg["model_name"]

    def embed_query(self, texts):
        """Input is list of texts."""
        text_inputs = tokenizer(texts, padding=True, return_tensors="pt")
        text_features = clip.get_text_features(**text_inputs)
        return text_features

    def get_embedding_length(self):
        text_features = self.embed_query("sample_text")
        return text_features.shape[1]

    def get_image_embeddings(self, images):
        """Input is list of images."""
        image_inputs = processor(images=images, return_tensors="pt")
        image_features = clip.get_image_features(**image_inputs)
        return image_features

    def get_video_embeddings(self, frames_batch):
        """Input is list of list of frames in video."""
        self.batch_size = len(frames_batch)
        vid_embs = []
        for frames in frames_batch:
            frame_embeddings = self.get_image_embeddings(frames)
            frame_embeddings = rearrange(frame_embeddings, "(b n) d -> b n d", b=len(frames_batch))
            # Normalize, mean aggregate and return normalized video_embeddings
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
            video_embeddings = frame_embeddings.mean(dim=1)
            video_embeddings = video_embeddings / video_embeddings.norm(dim=-1, keepdim=True)
            vid_embs.append(video_embeddings)
        return torch.cat(vid_embs, dim=0)
