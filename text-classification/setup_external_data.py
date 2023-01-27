"""Script that sets up external data for the text classification example app."""

import os
import sqlite3

import datasets

from inductor.data.table import sqlite


# Path to file in which sqlite database containing data should be stored
_DB_FILE_PATH = "./external_data.db"

# Maximum number of product reviews to be included
_MAX_NUM_REVIEWS = 10000


def _setup():
    """Downloads and stores relevant external data.

    Specifically, downloads and stores English-language product reviews and
    their associated metadata and star ratings from the
    [amazon_reviews_multi dataset](
    https://huggingface.co/datasets/amazon_reviews_multi) in a sqlite database
    table.  Removes any file that already exists at _DB_FILE_PATH.
    """
    if os.path.exists(_DB_FILE_PATH):
        os.remove(_DB_FILE_PATH)

    dataset = datasets.load_dataset(
        "amazon_reviews_multi", "en", split="train").shuffle(seed=42).select(
            range(_MAX_NUM_REVIEWS))
    review_rows = []
    for d in dataset:
        review_row = d.copy()
        del review_row["stars"]
        review_rows.append(review_row)

    conn = sqlite3.connect(_DB_FILE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reviews (language)")
    conn.commit()
    reviews_table = sqlite.SqliteKeyedTable(
        _DB_FILE_PATH, "reviews", "review_id", indexed_columns={
            "product_category", "product_id", "reviewer_id"})
    reviews_table.extend(review_rows)
    assert len(reviews_table) == len(review_rows)
    print(f"{len(review_rows)} reviews")
    print(f"Reviews table shape: {reviews_table.shape()}")


if __name__ == "__main__":
    _setup()
