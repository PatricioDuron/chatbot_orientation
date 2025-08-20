import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Path to the folder containing your PDFs
pdf_folder = os.path.join("data", "pdfs")

# List to store all extracted documents
all_documents = []

# Loop through all PDFs in the folder
for filename in tqdm(os.listdir(pdf_folder), desc="Processing PDFs"):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        try:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print(f"\n✅ Total documents extracted: {len(all_documents)}")

# Split the documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)
print(f"✅ Total chunks generated: {len(chunks)}")

# Save chunks as a JSON file
formatted_chunks = [
    {
        "content": chunk.page_content,
        "metadata": chunk.metadata
    }
    for chunk in chunks
]

output_path = os.path.join("data", "chunks.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(formatted_chunks, f, ensure_ascii=False, indent=2)

print(f"\n📁 Chunks saved to: {output_path}")
print("\n🧪 Example chunk preview:")
if chunks:
    print(f"Source: {chunks[0].metadata.get('source', 'unknown')}")
    print(chunks[0].page_content[:300], "...\n")
else:
    print("No chunks to preview.\n")
