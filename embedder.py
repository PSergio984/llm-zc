"""
embedder.py — Vector embedding wrapper using sentence-transformers.

Wraps the all-MiniLM-L6-v2 model to produce 384-dimensional normalized
vectors, suitable for cosine-similarity search.  Because the model's
output vectors are unit-length, the dot product of two vectors equals
their cosine similarity directly.
"""

from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model for encoding text
    and computing similarity scores.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialise the embedder with a model name.
        
        The default all-MiniLM-L6-v2 is compact (80 MB), fast on CPU,
        and produces good-quality embeddings for general English text.
        It outputs 384-dimensional unit vectors.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        """
        Encode a single text string (or a list of strings) into a
        numpy vector (or array of vectors).
        """
        return self.model.encode(text)

    def similarity(self, a, b):
        """
        Compute cosine similarity between two encoded vectors.
        A shortcut over sentence-transformers' built-in method.
        """
        return self.model.similarity(a, b)

    def dot_product(self, a, b):
        """
        Compute the dot product of two vectors.
        
        For unit vectors (which all-MiniLM-L6-v2 produces) this is
        equivalent to cosine similarity.
        """
        return a.dot(b)
