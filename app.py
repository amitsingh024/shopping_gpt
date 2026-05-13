import streamlit as st

from utils.loader import load_products
from utils.embedder import generate_embeddings
from utils.retriever import ProductRetriever
from utils.reranker import rerank_products
from utils.prompts import SHOPPING_PROMPT
from utils.generator import generate_response

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

st.set_page_config(page_title="ShoppingGPT")

st.title("🛍️ ShoppingGPT")
st.caption("AI shopping assistant powered by semantic search")

products_df = load_products("products.csv")

embeddings = generate_embeddings(
    products_df["combined_text"].tolist()
)

retriever = ProductRetriever(
    embeddings,
    products_df
)

query = st.text_input(
    "What are you looking for?",
    placeholder="Best headphones for work meetings under 30000"
)

if query:

    query_embedding = embedding_model.encode(query)

    results = retriever.search(
        query_embedding
    )

    reranked = rerank_products(results)

    ai_response = generate_response(
        query,
        reranked,
        SHOPPING_PROMPT
    )

    st.subheader("AI Recommendation")
    st.write(ai_response)

    st.subheader("Recommended Products")

    for _, row in reranked.iterrows():

            st.write(row['description'])