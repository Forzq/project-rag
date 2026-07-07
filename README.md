# project-rag

## Project structure

Core reusable code lives in `src/rag_local/`:

```text
chunking.py      # fixed, recursive, semantic chunking logic
chunk_io.py      # read/write chunk files
embeddings.py    # OpenRouter and HuggingFace embedders
vector_store.py  # ChromaDB loading and retrieval helpers
config.py        # shared model names and paths
```

Files in `scripts/` are thin CLI entry points that call this shared code.
The FastAPI app in `app.py` uses the same shared retrieval and embedding classes.

## PDF text extraction

Install dependencies:

```powershell
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Print PDF text to console:

```powershell
python scripts\extract_pdf_text.py path\to\file.pdf
```

Save PDF text to a `.txt` file:

```powershell
python scripts\extract_pdf_text.py path\to\file.pdf -o output\file.txt
```

## Complex PDF parsing with LlamaParse

Use this for PDFs with tables, columns, scanned pages, or complex visual layout.
The script sends the PDF to LlamaCloud and saves structured Markdown.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
LLAMA_CLOUD_API_KEY=llx-your-key
```

Parse a complex PDF:

```powershell
python scripts\extract_complex_pdf_llamaparse.py path\to\file.pdf -o output\file.md
```

For harder documents, use the stronger tier:

```powershell
python scripts\extract_complex_pdf_llamaparse.py path\to\file.pdf -o output\file.md --tier agentic_plus
```

## Chunking

Run all three chunking strategies:

```powershell
python scripts\chunk_all.py md_output\my_document.md
```

Semantic chunking uses OpenRouter embeddings. Add this to `.env` before running it:

```env
OPENROUTER_API_KEY=sk-or-your-key
```

Default embedding model:

```text
qwen/qwen3-embedding-4b
```

Run each strategy separately:

```powershell
python scripts\chunk_fixed_size.py md_output\my_document.md -o chunks_output\fixed_size_chunks.md
python scripts\chunk_recursive.py md_output\my_document.md -o chunks_output\recursive_chunks.md
python scripts\chunk_semantic.py md_output\my_document.md -o chunks_output\semantic_chunks.md
```

Results are saved as separate Markdown files in `chunks_output/` for manual comparison.

## Embeddings

Generate embeddings for an existing chunks file:

```powershell
python scripts\embed_chunks.py chunks_output\harry_potter_1_embedding\semantic_chunks.md --output-dir embeddings_output\harry_potter_1_semantic
```

The script saves:

```text
embeddings_output\harry_potter_1_semantic\openai_text_embedding_3_small.npy
embeddings_output\harry_potter_1_semantic\intfloat_e5_large_v2.npy
embeddings_output\harry_potter_1_semantic\chunks_metadata.json
```

## ChromaDB

Load existing chunks and embeddings into local persistent ChromaDB:

```powershell
python scripts\load_chroma.py --collection harry_potter_openai_1536 --embeddings embeddings_output\harry_potter_1_semantic\openai_text_embedding_3_small.npy --metadata embeddings_output\harry_potter_1_semantic\chunks_metadata.json --reset
```

The local Chroma database is stored in `chroma_db/`.

## Basic Retrieval UI

Run the local FastAPI app:

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The app embeds your query with `openai/text-embedding-3-small`, searches the
`harry_potter_openai_1536` Chroma collection with cosine similarity, and shows
the top-k chunks for visual relevance review.
