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
