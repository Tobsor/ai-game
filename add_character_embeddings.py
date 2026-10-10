"""Compatibility entrypoint for the complete Word/CSV ingestion pipeline."""
from sync_data import run


def add_character_embeddings():
    return run(sync_embeddings=True)


def main():
    add_character_embeddings()


if __name__ == "__main__":
    main()
