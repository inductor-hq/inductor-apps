"""Basic intro example app.

An example app illustrating Inductor's core abstractions.
"""

import datetime

import inductor


# Construct App object
app = inductor.App(
    inductor.env.Local("app_data"))

# Add an existing external data table
# (Per setup_external_data.py, this table contains a single column
# named "entity".)
app.tables.add_existing_sqlite_table(
    "entities",
    "./external_data.db",
    sqlite_table_name="entities")

# Create a new table to store app-generated data
app.tables.create_table(
    "history",
    faster_query_columns=["said_hello_at"])


@app("/")
def hello_world():
    p = app.Page("Hello World")

    p.heading("Hello World example app")
    
    name = p.select(
        app.tables["entities"].select("entity").values() + ["a custom name"],
        label="Say hello to")
    if name == "a custom name":
        name = p.input("Enter a name")

    if p.button("Say hello!") and name is not None:
        app.tables["history"].append({
            "name": name, "said_hello_at": datetime.datetime.now()})
        p.print(f"Hello {name}!")

    if len(app.tables["history"]) > 0:
        p.subsubheading("Your history of saying hello")
        with p.columns():
            with p.column():
                p.histogram(
                    app.tables["history"].select("name").values(),
                    title="Name counts")
            with p.column():
                p.data_table(
                    app.tables["history"].select(
                        "*", "ORDER BY said_hello_at DESC"))

    return p
