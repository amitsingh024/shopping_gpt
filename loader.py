import pandas as pd


def load_products(file_path):
    df = pd.read_csv(file_path)

    df["combined_text"] = (
        df["product_name"] + " " +
        df["category"] + " " +
        df["brand"] + " " +
        df["description"]
    )

    return df