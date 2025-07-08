# tests/test_ingestion.py
import tempfile
from pathlib import Path
from app.ingestion import load_and_index, get_vectorstore, set_vectorstore

def test_load_and_index_text_file():
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("This is a test document. It contains some text for embedding.")
        temp_file_path = Path(f.name)

    # Ensure vectorstore is None before starting
    set_vectorstore(None)

    # Load and index this file
    load_and_index(temp_file_path)

    vs = get_vectorstore()
    assert vs is not None

    # Clean up temp file after test
    temp_file_path.unlink()
