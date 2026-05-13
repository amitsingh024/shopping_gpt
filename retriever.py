import faiss
import numpy as np


class ProductRetriever:

    def __init__(self, embeddings, products_df):
        self.products_df = products_df

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]),
            top_k
        )

        results = self.products_df.iloc[
            indices[0]
        ]

        return results