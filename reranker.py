
def rerank_products(results_df):
    ranked = results_df.sort_values(
        by=["rating"],
        ascending=False
    )

    return ranked