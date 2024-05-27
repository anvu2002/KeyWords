"""
Transform Sentences & paragraphs to a 384 dimensional dense vector space
Usage: 
- Clustering
- Semantic Searches
- In this one: Finding the most matching picture (based on the caption)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

t2v_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def text_to_vec(text: str) -> np.array:
    embeddings = np.array(t2v_model.encode(text))

    return embeddings


prompt = text_to_vec("Hacking is the art of deceiving")

image_text_1 = text_to_vec("a person wearing black hoodie and starting an SSH handshake")

image_text_2= text_to_vec("a person with black hoodie and replaying an SSH handshake")


# print(f"prompt = {prompt}\n,image_text_1 = {image_text_1}\n,image_text_2 = {image_text_2}\n")

print(f"image_text_1 = {float(np.linalg.norm(image_text_1-prompt))}")
print(f"image_text_2 = {float(np.linalg.norm(image_text_2-prompt))}")

breakpoint()
