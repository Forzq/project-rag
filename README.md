# project-rag

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
