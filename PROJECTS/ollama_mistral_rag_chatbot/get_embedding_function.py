"""
from langchain.embeddings import SentenceTransformerEmbeddings

def get_embedding_function():
    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    return embeddings
"""
from sentence_transformers import SentenceTransformer

def get_embedding_function():
    # Initialize the SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Define a function to generate embeddings
    def embed_text(text):
        # Use the encode method to generate embeddings
        return model.encode(text, convert_to_tensor=True)

    return embed_text


