"""Docling conversion microservice — accepts file uploads, returns markdown."""

import asyncio
import logging
import os
import tempfile
import threading
import time
from functools import partial

import pymupdf
import pymupdf4llm
import pypdfium2 as pdfium
import torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("docling-service")

app = FastAPI(title="Docling Converter")


def _resolve_device() -> AcceleratorDevice:
    env = (os.environ.get("DOCLING_DEVICE") or "auto").lower()
    if env == "cpu":
        return AcceleratorDevice.CPU
    if env.startswith("cuda") and torch.cuda.is_available():
        return AcceleratorDevice.CUDA
    if env == "auto" and torch.cuda.is_available():
        return AcceleratorDevice.CUDA
    return AcceleratorDevice.CPU


DEVICE = _resolve_device()
NUM_THREADS = int(os.environ.get("DOCLING_NUM_THREADS") or os.cpu_count() or 4)
log.info("accelerator device=%s num_threads=%d", DEVICE.value, NUM_THREADS)


def _build_converter(do_ocr: bool, do_table_structure: bool) -> DocumentConverter:
    opts = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
        accelerator_options=AcceleratorOptions(device=DEVICE, num_threads=NUM_THREADS),
    )
    if do_table_structure:
        # Fast mode uses a smaller TableFormer variant — roughly 2x faster, small accuracy hit.
        opts.table_structure_options.mode = TableFormerMode.FAST
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


# Pre-build four variants so first request of each kind doesn't pay model-load cost.
# Each converter lazy-loads its models on first use; we warm the common one explicitly below.
_CONVERTERS: dict[tuple[bool, bool], DocumentConverter] = {
    (False, True): _build_converter(do_ocr=False, do_table_structure=True),
    (True, True): _build_converter(do_ocr=True, do_table_structure=True),
    (False, False): _build_converter(do_ocr=False, do_table_structure=False),
    (True, False): _build_converter(do_ocr=True, do_table_structure=False),
}


def _probe_pdf(pdf_path: str, sample_pages: int = 5, min_chars: int = 100) -> tuple[int, bool]:
    """Return (page_count, has_text_layer)."""
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        total = len(pdf)
        if total == 0:
            return 0, False
        step = max(1, total // sample_pages)
        hits = 0
        checked = 0
        for i in range(0, total, step):
            page = pdf[i]
            text = page.get_textpage().get_text_range()
            checked += 1
            if len(text.strip()) >= min_chars:
                hits += 1
            if checked >= sample_pages:
                break
        return total, hits >= max(1, checked // 2 + 1)
    except Exception as e:
        log.warning("pdf probe failed: %s — assuming OCR needed", e)
        return 0, False


class _Heartbeat:
    """Logs progress every N seconds while a long-running job is in flight."""

    def __init__(self, label: str, pages: int, interval: float = 5.0):
        self.label = label
        self.pages = pages
        self.interval = interval
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = time.perf_counter() - self._t0
            if self.pages > 0:
                log.info(
                    "[%s] progress: elapsed=%.1fs (%.2fs/page so far, budget %.0fs at current pace)",
                    self.label, elapsed, elapsed / self.pages, elapsed,
                )
            else:
                log.info("[%s] progress: elapsed=%.1fs", self.label, elapsed)


def _convert_docling_sync(path: str, do_ocr: bool, do_table_structure: bool, pages_expected: int, label: str) -> dict:
    converter = _CONVERTERS[(do_ocr, do_table_structure)]
    log.info(
        "[%s] starting docling: pages≈%d ocr=%s tables=%s device=%s threads=%d",
        label, pages_expected, do_ocr, do_table_structure, DEVICE.value, NUM_THREADS,
    )
    t0 = time.perf_counter()
    with _Heartbeat(label, pages_expected):
        result = converter.convert(path)
    t_convert = time.perf_counter() - t0
    markdown = result.document.export_to_markdown()
    elapsed = time.perf_counter() - t0
    pages = len(result.document.pages)
    log.info(
        "[%s] DONE docling pages=%d elapsed=%.1fs convert=%.1fs export=%.1fs → %.2f pages/sec (%.2fs/page)",
        label, pages, elapsed, t_convert, elapsed - t_convert,
        pages / max(elapsed, 1e-9), elapsed / max(pages, 1),
    )
    return {
        "markdown": markdown,
        "pages": pages,
        "elapsed_seconds": round(elapsed, 2),
        "pages_per_second": round(pages / max(elapsed, 1e-9), 2),
        "engine": "docling",
        "ocr_used": do_ocr,
        "device": DEVICE.value,
    }


def _convert_pymupdf_sync(path: str, pages_expected: int, label: str) -> dict:
    log.info("[%s] starting pymupdf4llm: pages≈%d (text-layer extraction, no OCR)", label, pages_expected)
    t0 = time.perf_counter()
    with _Heartbeat(label, pages_expected):
        markdown = pymupdf4llm.to_markdown(path, show_progress=False)
    elapsed = time.perf_counter() - t0
    pages = pages_expected
    log.info(
        "[%s] DONE pymupdf4llm pages=%d elapsed=%.1fs → %.2f pages/sec (%.3fs/page)",
        label, pages, elapsed,
        pages / max(elapsed, 1e-9), elapsed / max(pages, 1),
    )
    return {
        "markdown": markdown,
        "pages": pages,
        "elapsed_seconds": round(elapsed, 2),
        "pages_per_second": round(pages / max(elapsed, 1e-9), 2),
        "engine": "pymupdf",
        "ocr_used": False,
        "device": "cpu",
    }


def _convert_pymupdf_fast_sync(path: str, pages_expected: int, label: str) -> dict:
    """Ultra-fast PDF → Markdown using raw pymupdf.

    Extracts text per page with `get_text('dict')` and emits minimal markdown:
    lines whose dominant font size is significantly larger than the body-text
    size are promoted to headings. No table / image / code analysis.

    Typical speed: ~50–200 pages/sec on CPU.
    """
    log.info("[%s] starting pymupdf_fast: pages≈%d (raw text + heading heuristic)", label, pages_expected)
    t0 = time.perf_counter()
    with _Heartbeat(label, pages_expected):
        doc = pymupdf.open(path)
        total = len(doc)

        # First pass: sample font sizes from up to the first 20 pages to pick body-text size.
        from collections import Counter
        sample_counter: Counter = Counter()
        for i in range(min(20, total)):
            for b in doc[i].get_text("dict").get("blocks", []):
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        sample_counter[round(span["size"], 1)] += 1
        body_size = sample_counter.most_common(1)[0][0] if sample_counter else 10.0
        h1 = body_size * 1.6
        h2 = body_size * 1.3
        h3 = body_size * 1.15

        parts: list[str] = []
        for page in doc:
            for b in page.get_text("dict").get("blocks", []):
                block_lines: list[str] = []
                block_max_size = 0.0
                for line in b.get("lines", []):
                    text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if not text:
                        continue
                    line_max = max((s.get("size", body_size) for s in line.get("spans", [])), default=body_size)
                    block_max_size = max(block_max_size, line_max)
                    block_lines.append(text)
                if not block_lines:
                    continue
                block_text = " ".join(block_lines)
                if block_max_size >= h1:
                    parts.append(f"# {block_text}")
                elif block_max_size >= h2:
                    parts.append(f"## {block_text}")
                elif block_max_size >= h3:
                    parts.append(f"### {block_text}")
                else:
                    parts.append(block_text)
            parts.append("")  # blank line between pages
        markdown = "\n\n".join(parts)

    elapsed = time.perf_counter() - t0
    log.info(
        "[%s] DONE pymupdf_fast pages=%d body_size=%.1fpt elapsed=%.1fs → %.2f pages/sec",
        label, total, body_size, elapsed, total / max(elapsed, 1e-9),
    )
    return {
        "markdown": markdown,
        "pages": total,
        "elapsed_seconds": round(elapsed, 2),
        "pages_per_second": round(total / max(elapsed, 1e-9), 2),
        "engine": "pymupdf_fast",
        "ocr_used": False,
        "device": "cpu",
    }


@app.post("/convert")
async def convert_file(
    file: UploadFile = File(...),
    engine: str = Query(
        "auto",
        description="Conversion engine: 'pymupdf_fast' (ultra-fast raw text + "
        "heading heuristic), 'pymupdf' (fast, structured markdown), 'docling' "
        "(slow, OCR + layout + tables), or 'auto' (pymupdf for PDFs with text "
        "layer, docling otherwise).",
    ),
    force_ocr: bool = Query(False, description="(docling only) Force OCR even if PDF has a text layer"),
    skip_ocr: bool = Query(False, description="(docling only) Skip OCR unconditionally"),
    skip_tables: bool = Query(False, description="(docling only) Skip table structure detection"),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name
    size_mb = len(data) / (1024 * 1024)

    label = os.path.basename(file.filename or "upload")
    engine = (engine or "auto").lower()
    try:
        pages_expected = 0
        has_text = False
        if suffix.lower() == ".pdf":
            pages_expected, has_text = _probe_pdf(tmp_path)

        # Engine resolution.
        valid_engines = {"auto", "pymupdf_fast", "pymupdf", "docling"}
        if engine not in valid_engines:
            return JSONResponse(
                {"error": f"Unknown engine '{engine}'. Use one of: {sorted(valid_engines)}."},
                status_code=400,
            )
        if engine == "auto":
            engine = "pymupdf" if (suffix.lower() == ".pdf" and has_text) else "docling"

        # pymupdf engines only support PDFs — fall back to docling for other formats.
        if engine in {"pymupdf", "pymupdf_fast"} and suffix.lower() != ".pdf":
            log.info("[%s] engine=%s requested for non-PDF; falling back to docling", label, engine)
            engine = "docling"

        log.info(
            "[%s] received: size=%.1fMB suffix=%s pages=%d text_layer=%s engine=%s",
            label, size_mb, suffix, pages_expected, has_text, engine,
        )

        loop = asyncio.get_event_loop()
        if engine == "pymupdf_fast":
            result = await loop.run_in_executor(
                None, partial(_convert_pymupdf_fast_sync, tmp_path, pages_expected, label)
            )
        elif engine == "pymupdf":
            result = await loop.run_in_executor(
                None, partial(_convert_pymupdf_sync, tmp_path, pages_expected, label)
            )
        else:
            if skip_ocr:
                do_ocr = False
            elif force_ocr:
                do_ocr = True
            else:
                do_ocr = not has_text if suffix.lower() == ".pdf" else False
            do_tables = not skip_tables
            result = await loop.run_in_executor(
                None, partial(_convert_docling_sync, tmp_path, do_ocr, do_tables, pages_expected, label)
            )
        return JSONResponse(result)
    except Exception as e:
        log.exception("[%s] conversion failed", label)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE.value,
        "cuda_available": torch.cuda.is_available(),
        "num_threads": NUM_THREADS,
    }
