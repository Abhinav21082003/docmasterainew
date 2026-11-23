import streamlit as st
import pdfplumber
import pytesseract
import numpy as np
import hashlib
import io
import re
import time
import os
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from google import genai

# Cache directory for persisting chunks/embeddings across sessions
CACHE_DIR = Path("docbot_cache")
CACHE_DIR.mkdir(exist_ok=True)


# =====================================================
# 2. STREAMLIT CONFIGURATION  (must be the first st call)
# =====================================================

st.set_page_config(
    page_title="DocMaster AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocMaster AI")
st.caption("Document Converter | OCR | RAG-based AI Assistant")


# =====================================================
# 1. CONFIGURATION
# =====================================================

def get_api_key():
    """Resolve the Gemini API key from Streamlit secrets or environment.

    Falls back to a hardcoded key only as a last resort (not recommended
    for production) and shows a warning.
    """
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key and key != "YOUR_GEMINI_API_KEY_HERE":
            return key
    except Exception:
        pass

    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    st.warning(
        "⚠️ Using a hardcoded API key. For production, set "
        "GEMINI_API_KEY in `.streamlit/secrets.toml` or as an environment "
        "variable."
    )
    return "AQ.Ab8RN6JoZn8Rp7HTK5Bl6OD3O_Bho4xgWgG4PQfshCnYJvZLMw"


GEMINI_API_KEY = get_api_key()

MODEL_NAME = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"


def detect_tesseract_cmd():
    """Auto-detect the Tesseract executable. Returns None if not found
    (e.g. on Linux cloud where Tesseract may not be installed)."""
    import shutil

    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        return env_path

    default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_win):
        return default_win

    shutil_path = shutil.which("tesseract")
    if shutil_path:
        return shutil_path

    return None


TESSERACT_CMD = detect_tesseract_cmd()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# RAG settings (overridable via the sidebar)
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40
TOP_K = 4
SIMILARITY_THRESHOLD = 0.30
MAX_FILE_SIZE_MB = 25
MAX_PDF_PAGES = 200

client = genai.Client(api_key=GEMINI_API_KEY)


# =====================================================
# 3. API RETRY FUNCTION
# =====================================================

def api_with_retry(function, retries=4, base_delay=1.0):
    """Retry an API call with exponential backoff + jitter.

    Retries on 429 (rate limit), 500/502/503 (server errors) and
    transient network failures. Raises the last error if all attempts fail.
    """

    last_error = None

    for attempt in range(retries):

        try:
            return function()

        except Exception as e:

            last_error = e
            error = str(e).lower()

            retryable = (
                "429" in error
                or "500" in error
                or "502" in error
                or "503" in error
                or "timeout" in error
                or "network" in error
                or "connection" in error
            )

            if retryable and attempt < retries - 1:

                delay = base_delay * (2 ** attempt)
                time.sleep(delay)

            else:
                raise

    raise last_error or Exception("API request failed after retries.")


def _preprocess_image_for_ocr(image):
    """Lightweight preprocessing to improve Tesseract accuracy."""
    try:
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.SHARPEN)
    except Exception:
        pass
    return image


def _render_pdf_page_to_image(page, dpi=200):
    """Render a pdfplumber page to a PIL Image for OCR fallback."""
    try:
        img_obj = page.to_image(resolution=dpi)
        pil_img = img_obj.original
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return pil_img
    except Exception:
        return None


def _ocr_pdf_pages(file_bytes, max_pages=MAX_PDF_PAGES):
    """OCR every page of a scanned/image-based PDF.

    Returns the concatenated text with page markers.
    """
    if not TESSERACT_CMD:
        raise RuntimeError(
            "Tesseract OCR is not installed on this server. "
            "Scanned PDFs cannot be processed without Tesseract."
        )

    import fitz  # PyMuPDF

    pages_text = []
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = min(len(pdf_document), max_pages)

    for page_number in range(total_pages):

        page = pdf_document[page_number]

        pix = page.get_pixmap(dpi=200)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image = _preprocess_image_for_ocr(image)

        page_text = pytesseract.image_to_string(image).strip()

        if page_text:
            pages_text.append(
                f"\nPage {page_number + 1}\n" + page_text + "\n"
            )

    pdf_document.close()
    return "".join(pages_text).strip()


def extract_pdf(file_bytes, force_ocr=False):
    """Extract text from a PDF.

    Tries pdfplumber first. If little/no text is found (scanned PDF),
    falls back to OCR on every page.
    """

    extracted_text = ""

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:

        pages = pdf.pages[:MAX_PDF_PAGES]

        for page_number, page in enumerate(pages, start=1):

            page_text = page.extract_text() or ""

            if page_text.strip():
                extracted_text += (
                    f"\nPage {page_number}\n"
                    + page_text.strip()
                    + "\n"
                )

    needs_ocr = force_ocr or len(extracted_text.strip()) < 50

    if needs_ocr:

        try:
            ocr_text = _ocr_pdf_pages(file_bytes)
            if ocr_text and len(ocr_text.strip()) > len(extracted_text.strip()):
                extracted_text = ocr_text
        except Exception as ocr_err:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    ocr_parts = []
                    for page_number, page in enumerate(
                        pdf.pages[:MAX_PDF_PAGES], start=1
                    ):
                        image = _render_pdf_page_to_image(page)
                        if image is None:
                            continue
                        image = _preprocess_image_for_ocr(image)
                        page_text = pytesseract.image_to_string(image).strip()
                        if page_text:
                            ocr_parts.append(
                                f"\nPage {page_number}\n"
                                + page_text
                                + "\n"
                            )
                    ocr_text = "".join(ocr_parts).strip()
                    if ocr_text:
                        extracted_text = ocr_text
            except Exception:
                if ocr_err:
                    raise ocr_err

    return extracted_text.strip()


def extract_image(file_bytes):
    """Extract text from an image file using OCR with preprocessing."""
    if not TESSERACT_CMD:
        raise RuntimeError(
            "Tesseract OCR is not installed on this server. "
            "Image OCR cannot be performed without Tesseract."
        )

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = _preprocess_image_for_ocr(image)
    return pytesseract.image_to_string(image).strip()


def extract_docx(file_bytes):
    document = Document(io.BytesIO(file_bytes))

    text = ""

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            text += paragraph.text + "\n"

    for table in document.tables:

        for row in table.rows:

            values = [
                cell.text.strip()
                for cell in row.cells
            ]

            text += " | ".join(values) + "\n"

    return text.strip()


def extract_txt(file_bytes):
    return file_bytes.decode(
        "utf-8",
        errors="replace"
    ).strip()


# =====================================================
# 5. DOCUMENT CONVERSION
# =====================================================

def create_txt(text):
    return text.encode("utf-8")


def create_word(text):
    document = Document()

    document.add_heading(
        "Extracted Document",
        level=1
    )

    for line in text.splitlines():
        document.add_paragraph(line)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def create_pdf(text):
    buffer = io.BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    x = 45
    y = height - 50
    line_height = 15

    pdf.setFont("Helvetica", 10)

    for line in text.splitlines():

        words = line.split()
        current_line = ""

        for word in words:

            test_line = (
                current_line + " " + word
            ).strip()

            if pdf.stringWidth(
                test_line,
                "Helvetica",
                10
            ) > width - 90:

                if y < 50:

                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = height - 50

                pdf.drawString(x, y, current_line)
                y -= line_height
                current_line = word

            else:
                current_line = test_line

        if y < 50:

            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - 50

        pdf.drawString(x, y, current_line)
        y -= line_height

    pdf.save()

    return buffer.getvalue()


def create_jpeg(text):
    font = None
    for candidate in [
        "arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
    ]:
        try:
            font = ImageFont.truetype(candidate, 18)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    lines = []

    for original_line in text.splitlines():

        line = original_line

        while len(line) > 100:
            lines.append(line[:100])
            line = line[100:]

        lines.append(line)

    if not lines:
        lines = ["No text available"]

    width = 1200
    line_height = 30
    height = max(400, len(lines) * line_height + 120)

    image = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(image)

    y = 60

    for line in lines:

        draw.text(
            (60, y),
            line,
            fill="black",
            font=font
        )

        y += line_height

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return buffer.getvalue()


# =====================================================
# 6. RAG - TEXT CHUNKING
# =====================================================

def _split_into_sentences(text):
    """Naive sentence splitter that keeps punctuation attached."""
    sentence_enders = re.compile(r"(?<=[.!?])\s+")
    parts = sentence_enders.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into sentence-aware chunks.

    Args:
        text: Full document text (may contain ``Page N`` markers).
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of overlapping words between chunks.

    Returns:
        List of chunk dicts: {"text": str, "page": int|None}
    """

    sentences = _split_into_sentences(text)

    chunks = []
    current_words = []
    current_page = None

    def flush():
        nonlocal current_words, current_page
        if current_words:
            chunks.append({
                "text": " ".join(current_words),
                "page": current_page,
            })
        current_words = []
        current_page = None

    for sentence in sentences:

        sentence_words = sentence.split()

        sentence_page = current_page
        for w in sentence_words:
            m = re.match(r"Page\s*(\d+)", w)
            if m:
                sentence_page = int(m.group(1))

        if current_words and len(current_words) + len(sentence_words) > chunk_size:
            flush()

        current_words.extend(sentence_words)
        current_page = sentence_page

        if len(current_words) >= chunk_size:
            flush()

    flush()

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1]["text"].split()
            tail = (
                prev_words[-chunk_overlap:]
                if len(prev_words) >= chunk_overlap
                else prev_words
            )
            new_text = " ".join(tail) + " " + chunks[i]["text"]
            overlapped.append({
                "text": new_text,
                "page": chunks[i]["page"],
            })
        chunks = overlapped

    return chunks


# =====================================================
# 7. RAG - DOCUMENT EMBEDDINGS + PERSISTENCE
# =====================================================

def _cache_paths(file_hash):
    """Return paths for cached chunks and embeddings."""
    return (
        CACHE_DIR / f"{file_hash}_chunks.json",
        CACHE_DIR / f"{file_hash}_embeddings.npz",
    )


def save_to_cache(file_hash, chunks, embeddings):
    chunks_path, emb_path = _cache_paths(file_hash)
    serializable = [{"text": c["text"], "page": c.get("page")} for c in chunks]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    np.savez(emb_path, embeddings=embeddings)


def load_from_cache(file_hash):
    chunks_path, emb_path = _cache_paths(file_hash)
    if not chunks_path.exists() or not emb_path.exists():
        return None
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        data = np.load(emb_path)
        embeddings = data["embeddings"]
        return chunks, embeddings
    except Exception:
        return None


def generate_embeddings(chunks, file_hash=None):
    """Generate (or load from cache) normalized embeddings for chunks."""

    if file_hash:
        cached = load_from_cache(file_hash)
        if cached is not None:
            return cached[1]

    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]

    all_embeddings = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = api_with_retry(
            lambda: client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config={
                    "task_type": "RETRIEVAL_DOCUMENT"
                }
            )
        )
        batch_embeddings = [
            item.values
            for item in response.embeddings
        ]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(
        all_embeddings,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
    embeddings = embeddings / (norms + 1e-10)

    if file_hash:
        save_to_cache(file_hash, chunks, embeddings)

    return embeddings


# =====================================================
# 8. RAG - QUESTION EMBEDDING
# =====================================================

def generate_query_embedding(question):

    response = api_with_retry(
        lambda: client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
            config={
                "task_type": "RETRIEVAL_QUERY"
            }
        )
    )

    embedding = np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    return embedding / (norm + 1e-10)


# =====================================================
# 9. RAG - SEMANTIC SEARCH
# =====================================================

def retrieve_relevant_chunks(
    question,
    chunks,
    embeddings,
    top_k=TOP_K,
    threshold=SIMILARITY_THRESHOLD,
):

    if len(chunks) == 0 or embeddings is None or len(embeddings) == 0:
        return []

    query_embedding = generate_query_embedding(question)

    similarities = embeddings @ query_embedding

    top_indices = np.argsort(similarities)[::-1]

    results = []
    for index in top_indices:

        if index >= len(chunks):
            break

        score = float(similarities[index])
        if score < threshold:
            break

        chunk = chunks[index]
        results.append({
            "chunk_number": int(index) + 1,
            "text": chunk["text"] if isinstance(chunk, dict) else chunk,
            "page": chunk.get("page") if isinstance(chunk, dict) else None,
            "score": score,
        })

        if len(results) >= top_k:
            break

    return results


# =====================================================
# 10. RAG - GEMINI ANSWER GENERATION
# =====================================================

def generate_rag_answer(question, relevant_chunks):

    if not relevant_chunks:
        return (
            "I couldn't find any relevant content in the document to answer "
            "your question. Please try rephrasing or upload a document that "
            "covers this topic."
        )

    context_parts = []
    for item in relevant_chunks:
        page_ref = f" (page {item['page']})" if item.get("page") else ""
        context_parts.append(
            f"\n[Chunk {item['chunk_number']}{page_ref}]\n"
            f"{item['text']}\n"
        )

    context = "".join(context_parts)

    prompt = f"""
You are DocMaster AI, an intelligent document assistant.

Answer the user's question using the supplied document context.

Instructions:

1. Use only the retrieved context to answer.
2. Do not invent information.
3. If the answer is not present in the context,
   clearly say that it was not found in the document.
4. Give a clear and easy-to-understand answer.
5. Use bullet points when useful.
6. Reference the chunk/page number when citing facts.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = api_with_retry(
        lambda: client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    )

    return response.text or "No answer was generated."


# =====================================================
# 11. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ DocMaster Settings")

st.sidebar.markdown("""
### Supported Formats

- PDF
- PNG
- JPG / JPEG
- TXT
- DOCX

### AI Features

- Gemini 3.5 Flash Lite
- RAG pipeline
- Semantic search
- Document embeddings
- OCR (incl. scanned PDFs)
""")

# ---- Runtime configuration controls ----
st.sidebar.divider()
st.sidebar.subheader("RAG Configuration")

chunk_size = st.sidebar.number_input(
    "Chunk size (words)",
    min_value=50,
    max_value=2000,
    value=st.session_state.get("cfg_chunk_size", CHUNK_SIZE),
    step=50,
    help="Number of words per chunk.",
)
chunk_overlap = st.sidebar.number_input(
    "Chunk overlap (words)",
    min_value=0,
    max_value=500,
    value=st.session_state.get("cfg_chunk_overlap", CHUNK_OVERLAP),
    step=10,
    help="Overlapping words between adjacent chunks.",
)
top_k = st.sidebar.number_input(
    "Top-K chunks to retrieve",
    min_value=1,
    max_value=20,
    value=st.session_state.get("cfg_top_k", TOP_K),
    step=1,
)
similarity_threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.get("cfg_similarity_threshold", SIMILARITY_THRESHOLD),
    step=0.05,
    help="Minimum cosine similarity for a chunk to be retrieved.",
)

# Persist config into session state so processing uses them
st.session_state.cfg_chunk_size = chunk_size
st.session_state.cfg_chunk_overlap = chunk_overlap
st.session_state.cfg_top_k = top_k
st.session_state.cfg_similarity_threshold = similarity_threshold

# Model selection
model_option = st.sidebar.selectbox(
    "Generation model",
    options=["gemini-3.5-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
    index=0,
)
st.session_state.cfg_model = model_option

st.sidebar.info(
    "Chunk embeddings are cached to disk (docbot_cache/) and reused "
    "across sessions for the same document."
)

# Tesseract status
tesseract_ok = os.path.exists(TESSERACT_CMD)
if tesseract_ok:
    st.sidebar.success(f"✅ Tesseract: {TESSERACT_CMD}")
else:
    st.sidebar.error(
        f"❌ Tesseract not found at {TESSERACT_CMD}. "
        "Install Tesseract OCR or set TESSERACT_CMD env var."
    )


# =====================================================
# 12. FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "png", "jpg", "jpeg", "txt", "docx"],
    help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB",
)


# =====================================================
# 13. DOCUMENT PROCESSING
# =====================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"File too large ({file_size_mb:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )
        st.stop()

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    file_name = re.sub(
        r'[\\/*?:"<>|]',
        "_",
        uploaded_file.name.rsplit(".", 1)[0]
    )

    file_extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    st.success(f"Uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")

    # Use session-scoped config (fall back to module constants)
    chunk_size = st.session_state.get("cfg_chunk_size", CHUNK_SIZE)
    chunk_overlap = st.session_state.get("cfg_chunk_overlap", CHUNK_OVERLAP)
    top_k = st.session_state.get("cfg_top_k", TOP_K)
    similarity_threshold = st.session_state.get(
        "cfg_similarity_threshold", SIMILARITY_THRESHOLD
    )
    model_name = st.session_state.get("cfg_model", MODEL_NAME)

    extracted_text = None  # safety: defined before the branch below

    # Check whether this document was already processed (cache or session)
    cached = load_from_cache(file_hash)
    session_match = st.session_state.get("document_hash") == file_hash

    if cached is not None and not session_match:
        # Restore from disk cache
        chunks, embeddings = cached
        st.session_state.document_chunks = chunks
        st.session_state.document_embeddings = embeddings
        st.session_state.document_hash = file_hash
        st.session_state.chat_history = []
        st.session_state.extracted_text = "\n".join(c["text"] for c in chunks)
        st.success(f"Restored from cache! {len(chunks)} chunks loaded.")
    elif not session_match:
        # Clear previous document data
        st.session_state.pop("document_chunks", None)
        st.session_state.pop("document_embeddings", None)
        st.session_state.pop("extracted_text", None)
        st.session_state.pop("chat_history", None)

        try:

            # -------------------------------
            # Extract document text
            # -------------------------------

            with st.spinner("Extracting document text..."):

                if file_extension == "pdf":
                    force_ocr = st.session_state.get("force_ocr_pdf", False)
                    extracted_text = extract_pdf(
                        file_bytes,
                        force_ocr=force_ocr
                    )

                elif file_extension in ["png", "jpg", "jpeg"]:
                    extracted_text = extract_image(file_bytes)

                elif file_extension == "txt":
                    extracted_text = extract_txt(file_bytes)

                elif file_extension == "docx":
                    extracted_text = extract_docx(file_bytes)

                else:
                    extracted_text = ""

            if not extracted_text:
                st.error(
                    "No text found in the document. "
                    "Scanned PDFs require OCR — ensure Tesseract is installed."
                )
                st.stop()

            # -------------------------------
            # Create RAG chunks
            # -------------------------------

            with st.spinner("Splitting document into chunks..."):
                chunks = split_into_chunks(
                    extracted_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            # -------------------------------
            # Generate embeddings
            # -------------------------------

            with st.spinner(
                f"Generating embeddings for {len(chunks)} chunks..."
            ):
                embeddings = generate_embeddings(
                    chunks,
                    file_hash=file_hash,
                )

            # Save everything in session state
            st.session_state.document_chunks = chunks
            st.session_state.document_embeddings = embeddings
            st.session_state.extracted_text = extracted_text
            st.session_state.document_hash = file_hash
            st.session_state.chat_history = []

            st.success(
                f"RAG indexing complete! "
                f"{len(chunks)} chunks created."
            )

        except Exception as e:
            st.error(f"Document processing error: {str(e)}")
            st.stop()

    else:
        # Already in session state
        extracted_text = st.session_state.extracted_text
        chunks = st.session_state.document_chunks
        embeddings = st.session_state.document_embeddings

    # Guard: if extracted_text is still None, something went wrong
    if extracted_text is None:
        st.error("Unable to extract text from the document. Please try again.")
        st.stop()

    # =================================================
    # 14. DISPLAY EXTRACTED TEXT
    # =================================================

    st.subheader("📃 Extracted Document Text")

    with st.expander("View extracted text"):

        st.text_area(
            "Extracted Content",
            value=extracted_text,
            height=250
        )

    # =================================================
    # 15. DOWNLOAD OPTIONS
    # =================================================

    st.subheader("⬇️ Download Converted Documents")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "Download TXT",
            data=create_txt(extracted_text),
            file_name=f"{file_name}.txt",
            mime="text/plain"
        )

        st.download_button(
            "Download PDF",
            data=create_pdf(extracted_text),
            file_name=f"{file_name}.pdf",
            mime="application/pdf"
        )

    with col2:

        st.download_button(
            "Download Word",
            data=create_word(extracted_text),
            file_name=f"{file_name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        st.download_button(
            "Download JPEG",
            data=create_jpeg(extracted_text),
            file_name=f"{file_name}.jpg",
            mime="image/jpeg"
        )

    # =================================================
    # 16. RAG CHATBOT
    # =================================================

    st.divider()

    st.subheader("🤖 Ask Your Document - RAG Chatbot")

    st.write(
        f"Indexed chunks: {len(chunks)}"
    )

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    # Clear chat button
    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

    # OCR toggle for scanned PDFs
    if file_extension == "pdf":
        st.checkbox(
            "Force OCR on all PDF pages (useful for scanned PDFs)",
            key="force_ocr_pdf",
            value=st.session_state.get("force_ocr_pdf", False),
            help="Enable if the PDF is image-based and text extraction returned little content.",
        )

    # Display chat history
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # User input
    question = st.chat_input(
        "Ask anything about your uploaded document..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                # ---------------------------
                # RETRIEVAL
                # ---------------------------

                with st.spinner(
                    "Searching relevant document chunks..."
                ):

                    relevant_chunks = retrieve_relevant_chunks(
                        question,
                        st.session_state.document_chunks,
                        st.session_state.document_embeddings,
                        top_k=top_k,
                        threshold=similarity_threshold,
                    )

                if not relevant_chunks:
                    st.info(
                        "No relevant chunks found above the similarity threshold. "
                        "Try lowering the threshold in the sidebar."
                    )

                # ---------------------------
                # GENERATION
                # ---------------------------

                with st.spinner(
                    "Gemini is generating your answer..."
                ):

                    answer = generate_rag_answer(
                        question,
                        relevant_chunks
                    )

                st.markdown(answer)

                # Show retrieved chunks
                with st.expander(
                    "🔍 View retrieved document chunks"
                ):

                    for item in relevant_chunks:

                        page_ref = f" | Page {item['page']}" if item.get("page") else ""
                        st.write(
                            f"Chunk {item['chunk_number']}{page_ref} "
                            f"| Similarity: {item['score']:.4f}"
                        )

                        st.write(item["text"])

                        st.divider()

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"RAG chatbot error: {str(e)}"
                )

else:

    st.info(
        "Upload a document to start using DocMaster AI."
    )

    st.markdown("""
    ### 🚀 DocMaster AI Features

    1. Document text extraction
    2. Image OCR (with preprocessing)
    3. OCR fallback for scanned PDFs
    4. PDF to Word conversion
    5. Word to PDF conversion
    6. RAG-based document chatbot
    7. Semantic document search
    8. Gemini AI question answering
    """)
