import os
import re
import shutil
import chromadb
from pathlib import Path
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config & Credentials
# =============================================================================
AZURE_OPENAI_KEY         = os.getenv("AZURE_EMBEDDING_KEY")
AZURE_OPENAI_ENDPOINT    = "https://exl-services-resource.cognitiveservices.azure.com/"
AZURE_OPENAI_DEPLOYMENT  = "text-embedding-ada-002"
AZURE_OPENAI_API_VERSION = "2023-05-15"

CASE_STUDY_DIR   = "./Raw_casestudies/"
CHROMA_LOCAL_DIR = "./chroma_db/"
COLLECTION_NAME  = "exl_case_studies"

RESET = False  # set True to force a full rebuild

print("✅ Config loaded")
print(f"   Case study dir : {CASE_STUDY_DIR}")
print(f"   ChromaDB path  : {CHROMA_LOCAL_DIR}")


# =============================================================================
# Init Azure Embeddings + connection test
# =============================================================================
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

test_vector = embeddings.embed_query("test connection")
print(f"✅ Azure embeddings connected — vector dim: {len(test_vector)}")


# =============================================================================
# Load markdown files
# =============================================================================
def load_markdown_files(directory: str) -> list:
    files = []
    for path in Path(directory).glob("*.md"):
        content = path.read_text(encoding="utf-8")
        files.append({"filename": path.stem, "content": content})
    print(f"✅ Loaded {len(files)} markdown files from {directory}")
    return files

files = load_markdown_files(CASE_STUDY_DIR)


# =============================================================================
# Chunk by section
# =============================================================================
def chunk_by_section(filename: str, content: str) -> list:
    chunks = []
    sections = re.split(r'\n(?=## )', content)
    for section in sections:
        if not section.strip():
            continue
        first_line    = section.strip().split("\n")[0]
        section_label = first_line.replace("##", "").strip().lower()
        if "about" in section_label:
            continue
        chunks.append({
            "filename": filename,
            "section":  section_label,
            "text":     section.strip(),
        })
    return chunks

all_chunks = []
for file in files:
    chunks = chunk_by_section(file["filename"], file["content"])
    all_chunks.extend(chunks)
    print(f"   📄 {file['filename']} → {len(chunks)} chunks")

print(f"\n✅ Total chunks: {len(all_chunks)}")


# =============================================================================
# Init ChromaDB — build fresh or reuse existing
# =============================================================================
chroma_exists = Path(CHROMA_LOCAL_DIR).exists() and any(Path(CHROMA_LOCAL_DIR).iterdir())

if RESET and chroma_exists:
    print("🔄 RESET=True — clearing existing ChromaDB...")
    shutil.rmtree(CHROMA_LOCAL_DIR)
    chroma_exists = False

Path(CHROMA_LOCAL_DIR).mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_LOCAL_DIR,
    settings=Settings(anonymized_telemetry=False, allow_reset=True),
)

if RESET:
    chroma_client.reset()

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

print(f"✅ ChromaDB ready — existing vectors: {collection.count()}")


# =============================================================================
# Embed and insert (skips if collection already populated)
# =============================================================================
if collection.count() == 0:
    print(f"⚙️  Embedding {len(all_chunks)} chunks...")

    BATCH_SIZE = 50
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch   = all_chunks[i : i + BATCH_SIZE]
        texts   = [c["text"] for c in batch]
        ids     = [f"{c['filename']}__{c['section']}__{i + j}" for j, c in enumerate(batch)]
        metas   = [{"filename": c["filename"], "section": c["section"]} for c in batch]
        vectors = embeddings.embed_documents(texts)

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metas,
        )
        print(f"   ✅ Batch {i // BATCH_SIZE + 1} inserted ({len(batch)} chunks)")

    print(f"\n🎉 Embedding complete! Total vectors: {collection.count()}")
    print(f"   ChromaDB saved to {CHROMA_LOCAL_DIR}")

else:
    print(f"⏭️  Collection already has {collection.count()} vectors — skipping embedding")
    print(f"   Set RESET = True to rebuild from scratch")
