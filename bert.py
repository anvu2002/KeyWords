import transformers
import torch
import torch.nn.functional as F

# Load the BERT model and tokenizer
model = transformers.BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and encode the text
text = "This is a sample text for keyword extraction."
input_ids = tokenizer.encode(text, add_special_tokens=True)
input_ids_tensor = torch.tensor([input_ids])

# Use BERT to encode the meaning and context of the words and phrases in the text
with torch.no_grad():
    outputs = model(input_ids_tensor)

# Extract the hidden states
hidden_states = outputs.hidden_states  # List of tensors with shape [batch_size, seq_length, hidden_size]

# Get the embeddings from the last layer (shape: [seq_length, hidden_size])
last_layer_embeddings = hidden_states[-1][0]  # Removing the batch dimension

# Calculate the importance score for each token (e.g., using the norm of the embeddings)
token_importance = torch.norm(last_layer_embeddings, dim=1)

# Get the top 3 tokens based on their importance scores
top_indices = torch.topk(token_importance, 3).indices

# Decode the top tokens and print the top 3 keywords
top_keywords = [tokenizer.decode([input_ids[idx]]) for idx in top_indices]
print(top_keywords)
