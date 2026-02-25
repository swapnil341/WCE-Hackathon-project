pip install PyMuPDF langchain-text-splitters
from google.colab import files
import fitz  # PyMuPDF
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def parse_and_chunk_openstax():
    # 1. Load the document
    uploaded_files = files.upload() # This returns a dict like {'filename.pdf': b'content'}

    if not uploaded_files:
        raise ValueError("No file uploaded.")

    # Get the first (and likely only) filename from the uploaded files dictionary
    pdf_filename = list(uploaded_files.keys())[0]
    doc = fitz.open(pdf_filename) # Now doc is a PyMuPDF Document object

    # 2. Regex to detect OpenStax section headers (e.g., "1.2 History of Psychology")
    # This looks for a digit, a dot, a digit, a space, and then text.
    section_pattern = re.compile(r"^(\d+\.\d+)\s+(.+)")

    chunks = []
    current_section = "preface" # Default starting state
    chunk_id_counter = 1

    # 3. Setup the text splitter
    # 1000 characters with a 200 character overlap prevents cutting sentences in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )

    # 4. Iterate through every page in the PDF
    for page_num in range(len(doc)):
        page = doc[page_num]

        # .get_text("blocks") is crucial here. It reads the PDF structure,
        # preventing text from column A bleeding into column B.
        blocks = page.get_text("blocks")
        page_text_buffer = ""

        for block in blocks:
            # In PyMuPDF, block[4] contains the actual text string
            text = block[4].strip()
            if not text:
                continue

            # Check if the first line of this block is a new section header
            first_line = text.split('\n')[0].strip()
            match = section_pattern.match(first_line)

            if match:
                # Format section to match the hackathon's requested JSON style
                # (e.g., "2.2_approaches_to_research")
                section_num = match.group(1)
                section_title = match.group(2).replace(' ', '_').lower()
                # Clean up any special characters in the title
                section_title = re.sub(r'[^a-z0-9_]', '', section_title)

                current_section = f"{section_num}_{section_title}"

            page_text_buffer += text + "\n"

        # 5. Split the accumulated page text into standardized chunks
        if len(page_text_buffer.strip()) > 0:
            page_chunks = text_splitter.split_text(page_text_buffer)

            for chunk_text in page_chunks:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id_counter}",
                    "text": chunk_text,
                    "section": current_section,
                    "page_number": page_num + 1 # 1-indexed to match PDF page numbers
                })
                chunk_id_counter += 1

    return chunks

# --- Execution ---
# Replace with the actual path to your downloaded book.pdf
#pdf_file_path = "C:\Users\admin\Downloads\Psychology2e_WEB.pdf"
document_chunks = parse_and_chunk_openstax()

for i in range(5):
  document_chunks[i]
