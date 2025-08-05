# check_parser_output.py

import os
from src.components.parser import load_documents_with_docling

OUTPUT_DIR = "output"
OUTPUT_FILENAME = "parsed_document_chunks.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def inspect_parser_output():
    print("Running the document parser...")
    try:
        documents = load_documents_with_docling()
        
        print(f"Parser finished. Writing {len(documents)} chunks to '{output_filepath}' for inspection.")
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            for i, doc in enumerate(documents):
                # --- TEMPORARY CHANGE FOR DEBUGGING ---
                # Let's print the original metadata to the console to see what keys are available
                if i == 0: # We only need to see it once
                    print("\n--- ORIGINAL METADATA FOR FIRST CHUNK ---")
                    print(doc.metadata)
                    print("------------------------------------------\n")
                # --- END OF CHANGE ---

                f.write(f"==================== CHUNK {i + 1} ====================\n\n")
                # The metadata in the file will still show -1 for now
                f.write(f"METADATA:\n{doc.metadata}\n\n") 
                f.write("-------------------- CONTENT --------------------\n")
                f.write(f"{doc.page_content}\n\n\n")
        
        print(f"Successfully wrote output to {output_filepath}")
        print("Please check your CONSOLE to see the original metadata structure.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_parser_output()