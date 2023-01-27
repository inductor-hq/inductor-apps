"""Script that sets up external data for the basic intro example app."""

import os
import sqlite3

# Path to file in which sqlite database containing data should be stored
_DB_FILE_PATH = "./external_data.db"


def _setup():
    """Creates sqlite database."""
    if os.path.exists(_DB_FILE_PATH):
        os.remove(_DB_FILE_PATH)

    conn = sqlite3.connect(_DB_FILE_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS entities (entity)")
    for entity in ["world", "earth", "universe"]:
        conn.execute("INSERT INTO entities VALUES(?)", (entity,))
    conn.commit()


if __name__ == "__main__":
    _setup()
