FROM python:3.11-slim

WORKDIR /app

# System deps for docling (PDF processing, OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Keep HF/docling model caches at a stable, container-local path so they're
# part of the image layer (not re-downloaded on every container start).
# (RapidOCR always stores weights inside its own site-packages dir, so no env
# var needed for it.)
ENV HF_HOME=/opt/models/hf \
    DOCLING_ARTIFACTS_PATH=/opt/models/docling

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download docling models (layout + tableformer) into the image.
RUN docling-tools models download -o "$DOCLING_ARTIFACTS_PATH" layout tableformer

# Pre-download RapidOCR torch weights (det + cls + rec) so the container doesn't
# hit modelscope on first OCR request. Forcing the torch engine — the default
# onnxruntime engine isn't installed and we have torch anyway via docling.
RUN python -c "from rapidocr import RapidOCR, EngineType; \
    RapidOCR(params={'Det.engine_type': EngineType.TORCH, 'Cls.engine_type': EngineType.TORCH, 'Rec.engine_type': EngineType.TORCH})"

# PyMuPDF4LLM — fast engine for PDFs with a text layer (no OCR, no ML).
# Installed in a separate layer so adding/removing it does not invalidate the
# expensive model-download layers above.
# Capped <2: pymupdf4llm 2.0 is an empty placeholder package (the code moved
# into pymupdf 2.x) and does not install pymupdf. Quoted so the shell doesn't
# treat >/< as redirects.
RUN pip install --no-cache-dir "pymupdf4llm>=0.0.17,<2"

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
