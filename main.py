import os
import json
from src.ingestion.book_processor import parse_and_chunk_openstax


def main():
    # Path to the PDF file (provided by user in data/raw)
    pdf_file_path = os.path.join("data", "raw", "Psychology2e_WEB.pdf")

    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found at {pdf_file_path}")
        return

    print(f"Starting ingestion for: {pdf_file_path}")

    try:
        document_chunks = parse_and_chunk_openstax(pdf_file_path=pdf_file_path)

        chunk_count = len(document_chunks)
        print(f"Successfully chunked document into {chunk_count} chunks.")

        # Display first 5 chunks for verification
        print("\nVerifying first 5 chunks:")
        for i in range(min(5, chunk_count)):
            chunk = document_chunks[i]
            print(
                f"Chunk ID: {chunk['chunk_id']} | Section: {chunk['section']} | Page: {chunk['page_number']}")
            print(f"Text Preview: {chunk['text'][:100]}...")
            print("-" * 20)

        # Optional: Save processed chunks to data/processed
        output_path = os.path.join("data", "processed", "chunks.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(document_chunks, f, indent=4)
        print(f"\nProcessed chunks saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
