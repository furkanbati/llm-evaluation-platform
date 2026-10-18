import os

# Keep tests in memory: importing app.api must not create the default
# SQLite database file. Tests that need SQLite use a temporary path.
os.environ["DATABASE_PATH"] = ""
