# DocMasterAI — Document Conversion & RAG-based AI Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Streamlit web application for **document text extraction, OCR, file conversion, and AI-powered document chat** using Google's Gemini Generative AI. Upload PDFs, images, text, or Word documents — extract content, convert formats, and ask questions about your documents with a Retrieval-Augmented Generation (RAG) chatbot.

---

## 📖 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Requirements](#-requirements)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
- [Example Workflows](#-example-workflows)
- [Notes](#-notes)
- [About](#-about)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **File Upload** | Supports PDF, PNG, JPG, JPEG, TXT, and DOCX files (max 25 MB) |
| **Text Extraction** | Extract text from PDFs (`pdfplumber`), images (`pytesseract` + Tesseract OCR), Word docs, and plain text |
| **Scanned PDF OCR** | Automatic OCR fallback for image-based/scanned PDFs using PyMuPDF + Tesseract |
| **File Conversion** | Convert extracted text to TXT, PDF, Word (DOCX), and JPEG |
| **RAG Chatbot** | Ask questions about your document — answers generated via Gemini with retrieved context |
| **Semantic Search** | Cosine-similarity retrieval with adjustable Top-K and similarity threshold |
| **Citations** | Each answer references chunk number and page number for traceability |
| **Embedding Caching** | Chunks and embeddings cached to disk (`docbot_cache/`) — reused across sessions |
| **Sidebar Controls** | Tunable chunk size, overlap, Top-K, similarity threshold, and model selection |

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit | Web interface and file uploader |
| **Text Extraction** | `pdfplumber`, `pytesseract` | PDF text extraction and image OCR |
| **OCR Engine** | Tesseract-OCR | Optical character recognition for images & scanned PDFs |
| **PDF Rendering** | `PyMuPDF` (fitz) | Render PDF pages to images for OCR fallback |
| **Image Processing** | `Pillow` (PIL) | Image preprocessing (grayscale, autocontrast, sharpen) |
| **Word Docs** | `python-docx` | DOCX reading and writing |
| **PDF Generation** | `ReportLab` | Generate PDFs from extracted text |
| **AI / LLM** | Google Generative AI (Gemini) | Embeddings and chatbot answer generation |
| **Vector Math** | `NumPy` | Cosine similarity and embedding normalization |
| **Config** | Streamlit Secrets | Secure API key management |

---

## 📦 Requirements

### System Dependencies
- **Python 3.11+**
- **Tesseract OCR** — install from [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

### Python Libraries
Install with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
streamlit==1.67.0
pdfplumber==0.11.10
pytesseract==0.3.13
numpy==2.4.6
Pillow==13.0.0
python-docx==1.2.0
reportlab==5.1.0
google-genai==3.3.0
pymupdf==1.25.2
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Abhinav21082003/docmasterainew.git
cd docmasterainew
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

**Windows:** Download from [UB-Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki) and install to `C:\Program Files\Tesseract-OCR\`.

**Linux / Streamlit Cloud:** The `packages.txt` file includes `tesseract-ocr` for automatic system installation.

### 4. Configure API Key

Create `.streamlit/secrets.toml` in the project root:

```toml
GEMINI_API_KEY = "your-google-gemini-api-key"
```

Or set it as an environment variable:

```bash
export GEMINI_API_KEY="your-google-gemini-api-key"
```

> ⚠️ Never commit your API key to GitHub. Use Streamlit Secrets (cloud) or environment variables (local).

## 💻 Usage

### Run Locally

```bash
streamlit run doc.py
```

Then open http://localhost:8501 in your browser.

### Step-by-Step Workflow

1. **Upload** a document (PDF, PNG, JPG, JPEG, TXT, or DOCX)
2. **View** the extracted text in the expander
3. **Download** the extracted content as TXT, PDF, Word, or JPEG
4. **Ask questions** in the RAG chatbot — answers are generated from your document context with citations
5. **Tune retrieval** in the sidebar: adjust chunk size, overlap, Top-K, and similarity threshold

---

## ☁ Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo → main file: `doc.py`
4. In the app dashboard → **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-api-key"
   ```
5. Click **Deploy**

The `packages.txt` ensures Tesseract is installed on the cloud Linux environment.

---

## 📋 Example Workflows

### Uploading a PDF
- Text-based PDFs are extracted with `pdfplumber` page by page
- Scanned/image-based PDFs trigger automatic OCR via PyMuPDF + Tesseract
- Download extracted text as TXT, PDF, Word, or JPEG

### Uploading an Image
- Image is preprocessed (grayscale, autocontrast, sharpen) for better OCR accuracy
- Text extracted via `pytesseract`
- Download in TXT, PDF, Word, or JPEG formats

### Chatbot Interaction
1. Type a question in the chat input
2. The app retrieves the most relevant chunks using cosine similarity
3. Gemini generates an answer with chunk/page citations
4. Optionally download the chatbot response

---

## ⚠️ Notes

- **Tesseract path:** On Windows, Tesseract is auto-detected at `C:\Program Files\Tesseract-OCR\tesseract.exe`. On Linux/cloud, it must be installed via system packages (`packages.txt` includes `tesseract-ocr`).
- **API key:** Stored in `.streamlit/secrets.toml` — excluded from version control via `.gitignore`.
- **Embedding cache:** Stored in `docbot_cache/` — also excluded from version control.
- **File limits:** Max upload size is 25 MB; PDFs are capped at 200 pages for performance.

---

## 👤 About

**Author:** Abhinav21082003  
**Repository:** https://github.com/Abhinav21082003/docmasterainew  
**Visibility:** Private

Built with ❤️ using Streamlit and Google Gemini AI.
