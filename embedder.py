from sentence_transformers import SentenceTransformer


class Embedder:
    """Vector embedder using sentence-transformers with all-MiniLM-L6-v2.

    Produces 384-dim normalized vectors for cosine similarity search.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        return self.model.encode(text)

    def similarity(self, a, b):
        return self.model.similarity(a, b)

    def dot_product(self, a, b):
        return a.dot(b)
