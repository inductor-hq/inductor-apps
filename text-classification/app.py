"""Example demonstrating using Inductor to build a text classification app.

Demonstrates use of Inductor to build a text classification app, including a
custom data exploration and labeling tool, recurring model
(re)training, a prediction API, and a model exploration tool.
"""

import datetime
import random
from typing import Dict, List, Optional

import datasets
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import transformers

from inductor import ui
import inductor


# Construct App object
app = inductor.App(
    inductor.env.Local("app_data"))

# Add existing external data table
app.tables.add_existing_sqlite_table(
    "reviews",
    "./external_data.db",
    sqlite_table_name="reviews",
    primary_key_column="review_id")

# Configure navbar for all pages in this app
app.set_navbar(
    ui.format("Inlang", link_url="/"),
    [ui.format("Data Dashboard", link_url="/"),
     ui.format("Label Data", link_url="/label_data"),
     ui.format("Model Training", link_url="/model_training"),
     ui.format("Prediction Explorer", link_url="/prediction_explorer")])


# Data dashboard

@app("/")
def data_dashboard():
    p = app.Page("Data Dashboard")

    p.heading("Reviews")
    reviews = app.tables["reviews"]

    product_categories = p.multiselect(
        sorted(reviews.select("DISTINCT(product_category)").values()),
        label="Filter by product category:")
    product_categories_filter_clause = (
        "WHERE product_category IN ('" + "','".join(product_categories) + "')"
        if product_categories else "")

    p.data_table(
        reviews.select("*", product_categories_filter_clause + " LIMIT 2000"),
        out_of_row_count=reviews.select(
            "COUNT(*)", product_categories_filter_clause).value())

    hist_field_name = p.select(
        reviews.columns, label="Select a column to histogram")
    if hist_field_name:
        p.histogram(
            reviews.select(
                hist_field_name, product_categories_filter_clause).values(),
            xlabel=hist_field_name, ylabel="Number of reviews")

    if len(app.tables["labels"]) > 0:
        p.heading("Labels")
        p.print("Number of labeled reviews: " + str(len(app.tables["labels"])))
        p.data_table(app.tables["labels"])

    return p


# Data labeling

app.tables.create_keyed_table(
    "reviews_to_label", "review_id", faster_query_columns=["rank"])

@app(schedule="@up")
def populate_reviews_to_label():
    if len(app.tables["reviews_to_label"]) == 0:
        app.tables["reviews_to_label"].extend([
            {"review_id": r["review_id"], "rank": random.random()}
            for r in app.tables["reviews"]])


app.tables.create_keyed_table(
    "labels", "review_id", faster_query_columns=["label_time"])

@app("/label_data")
def label_data(review_id: Optional[str] = None):
    p = app.Page("Label Data")

    if not review_id:
        p.redirect(
            "/label_data?review_id=" +
            app.tables["reviews_to_label"].first_row()["review_id"])
        return p

    p.heading("Label Review")
    label = p.radio(
        ["Positive", "Neutral", "Negative", "Unclear"],
        horizontal=True, hot_keys=["1", "2", "3", "4"])
    if label:
        app.tables["labels"][review_id] = {
            "label": label, "label_time": datetime.datetime.now()}
        del app.tables["reviews_to_label"][review_id]
        p.redirect(
            "/label_data?review_id=" +
            app.tables["reviews_to_label"].first_row()["review_id"])
        return p
    review_row = app.tables["reviews"][review_id]
    p.print(review_row["review_title"])
    p.print(review_row["review_body"])
    p.print("Product category: " + review_row["product_category"])

    return p


# Model training

app.tables.create_keyed_table(
    "model_trainings", "id", faster_query_columns=["start_time"])

def embed_text_batch(
    pipe: transformers.Pipeline, strings: List[str]) -> List[np.ndarray]:
    embedding_matrix = pipe.model(**pipe.tokenizer(
        strings,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt")).last_hidden_state.detach().numpy().copy()
    return list(embedding_matrix.mean(axis=1))

@app(schedule="@daily")
def train_new_model():
    training_id = app.unique_id()
    app.tables["model_trainings"][training_id] = {
        "status": "preprocessing data", "start_time": datetime.datetime.now()}
    labels = app.tables["labels"].select(
        "review_id, label",
        "WHERE label != 'Unclear'").pandas_df().set_index("review_id")["label"]
    labels[labels == "Negative"] = 0
    labels[labels == "Neutral"] = 1
    labels[labels == "Positive"] = 2
    text = app.tables["reviews"].select(
        "review_id, review_title || ' <SEP> ' || review_body AS text",
        f"WHERE review_id IN {tuple(labels.index)}"
    ).pandas_df().set_index("review_id")
    pipe = transformers.pipeline(
        "feature-extraction", model="distilbert-base-cased")
    ds = datasets.Dataset.from_pandas(text.join(labels)).shuffle(seed=42)
    n = len(ds)
    n_train = int(0.8 * n)
    ds_train = ds.select(range(n_train)).map(
        lambda batch: {"embedding": embed_text_batch(pipe, batch["text"])},
        batched=True, batch_size=5)
    ds_test = ds.select(range(n_train, n)).map(
        lambda batch: {"embedding": embed_text_batch(pipe, batch["text"])},
        batched=True, batch_size=5)
    x_train = np.asarray(ds_train["embedding"])
    y_train = np.asarray(ds_train["label"])
    x_test = np.asarray(ds_test["embedding"])
    y_test = np.asarray(ds_test["label"])
    app.tables["model_trainings"].update(training_id, {
        "status": "training"})
    clf = LogisticRegressionCV(cv=2)
    clf.fit(x_train, y_train)
    acc_train = clf.score(x_train, y_train)
    acc_test = clf.score(x_test, y_test)
    app.models["clf"] = clf
    app.tables["model_trainings"].update(training_id, {
        "status": "complete",
        "acc_train": acc_train,
        "acc_test": acc_test,
        "end_time": datetime.datetime.now()})


@app("/model_training")
def model_training():
    p = app.Page("Model Training")
    p.heading("Model Trainings")
    if len(app.tables["model_trainings"]) == 0:
        p.print("You haven't trained any models yet.")
    label_counts = app.tables["labels"].select(
        "label, COUNT(*) as c",
        "WHERE label != 'Unclear' GROUP BY label").column_values()
    if min(label_counts["c"]) < 2 or len(label_counts["c"]) < 3:
        p.print("""
            In order to train a model, you must have at least two labeled
            reviews for each of 'Positive', 'Negative', and 'Neutral'.
            Please label more reviews.
        """)
        p.bar_chart(
            label_counts["label"], label_counts["c"],
            title="Current label counts")
    else:
        if p.button("Start new training"):
            train_new_model()
    if len(app.tables["model_trainings"]) > 0:
        p.data_table(app.tables["model_trainings"].select(
            "*", "ORDER BY start_time DESC"))
    p.print("Number of labeled reviews: " + str(len(app.tables["labels"])))
    return p


# Prediction API endpoint

@app("/predict")
def predict(review_title: str, review_body: str) -> Dict:
    text = review_title + " <SEP> " + review_body
    pipe = transformers.pipeline(
        "feature-extraction", model="distilbert-base-cased")
    probs = app.models["clf"].predict_proba(
        np.asarray(embed_text_batch(pipe, [text])))[0]
    return {
        "sentiment": ["Negative", "Neutral", "Positive"][probs.argmax()],
        "probability": probs.max()
    }


# Prediction explorer

@app("/prediction_explorer")
def prediction_explorer():
    p = app.Page("Prediction Explorer")
    p.heading("Prediction Explorer")
    p.print("Click on a row's arrow icon to view the model's prediction.")
    review = p.data_table(
        app.tables["reviews"].select("*", "LIMIT 2000"),
        out_of_row_count=len(app.tables["reviews"]),
        selectable=True)
    if review:
        review_title = review["review_title"]
        review_body = review["review_body"]
        prediction = predict(review_title, review_body)
        p.heading("Prediction")
        p.print("Sentiment: " + prediction["sentiment"])
        p.print("Probability: " + str(prediction["probability"]))
        p.print(f"Review title: {review_title}")
        p.print(f"Review body: {review_body}")
    return p
