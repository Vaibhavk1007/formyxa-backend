# app.py
# =============================================================
# Engine v2 – Vision Pipeline  +  DOCX Export Service
# Both run on ONE FastAPI server (port 8000)
#
# .env:
#   HANDW_API_BASE=http://localhost:8000
#   FLASK_DOCX_URL=http://localhost:8000/generate-docx
#   REDIS_URL=redis://...   ← ADD THIS (from Render Redis dashboard)
# =============================================================

import os
import base64
import json
import re
import unicodedata
import requests
import traceback
import io
import fitz          # PyMuPDF
import time
import numpy as np
import cv2
import redis as redis_lib          # ← CHANGE 1: added

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import RGBColor, Inches, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────

ENGINE_VERSION     = "v2.0.0"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o"
MAX_PDF_PAGES      = 20

OCR_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":  "application/json",
    "HTTP-Referer":  "http://localhost",
    "X-Title":       "Doc-Reconstructor-v2",
}

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

BASE_DIR      = os.path.dirname(__file__)
BASE_TEMPLATE = os.path.join(BASE_DIR, "base.docx")

API_KEY = os.getenv("HANDW_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "HANDW_API_KEY is NOT set. "
        "Start the server with HANDW_API_KEY environment variable."
    )


# ─────────────────────────────────────────────────────────────
# FASTAPI APP + MIDDLEWARE
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Handwritten-to-Doc Engine v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if request.url.path in ["/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)
    if request.url.path.startswith("/api") or request.url.path == "/generate-docx":
        key = request.headers.get("x-api-key")
        if not API_KEY or key != API_KEY:
            return JSONResponse(status_code=401, content={"error": "UNAUTHORIZED"})
    return await call_next(request)


# ─────────────────────────────────────────────────────────────
# CHANGE 2: JOB STORE — Redis (replaces in-memory dict)
# Survives cold starts and works across multiple Render instances.
# ─────────────────────────────────────────────────────────────

JOB_TTL = 60 * 60 * 3  # 3 hours — jobs auto-expire

def _get_redis():
    url = os.getenv("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL env var not set")
    return redis_lib.from_url(url, decode_responses=True)

def load_job(jobId: str):
    try:
        r = _get_redis()
        raw = r.get(f"job:{jobId}")
        return json.loads(raw) if raw else None
    except Exception as e:
        log("⚠️ Redis load_job error", repr(e))
        return None

def update_job(jobId: str, **updates):
    try:
        r = _get_redis()
        key = f"job:{jobId}"
        raw = r.get(key)
        existing = json.loads(raw) if raw else {"jobId": jobId}
        existing.update(updates)
        r.setex(key, JOB_TTL, json.dumps(existing))
    except Exception as e:
        log("⚠️ Redis update_job error", repr(e))


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

def log(step, data=None):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n🟦 [{ts}] STEP: {step}")
    if data is not None:
        print(data)


# =============================================================
# ░░░░  SECTION 1 — OCR / VISION PIPELINE  ░░░░░░░░░░░░░░░░░░
# =============================================================

def is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


def pdf_page_to_image_bytes(page) -> bytes:
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode PDF page as PNG")
    return buf.tobytes()


def pdf_to_image_bytes(pdf_bytes: bytes) -> bytes:
    doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = min(len(doc), MAX_PDF_PAGES)
    log("PDF pages to render", f"{total_pages} / {len(doc)}")

    page_images = []
    for i in range(total_pages):
        raw = pdf_page_to_image_bytes(doc.load_page(i))
        page_images.append(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))

    if len(page_images) == 1:
        stitched = page_images[0]
    else:
        max_w = max(img.shape[1] for img in page_images)
        padded = []
        for img in page_images:
            h, w = img.shape[:2]
            if w < max_w:
                img = np.hstack([img, np.ones((h, max_w - w, 3), dtype=np.uint8) * 255])
            padded.append(img)
        sep         = np.ones((10, max_w, 3), dtype=np.uint8) * 255
        interleaved = []
        for i, img in enumerate(padded):
            interleaved.append(img)
            if i < len(padded) - 1:
                interleaved.append(sep)
        stitched = np.vstack(interleaved)

    ok, buf = cv2.imencode(".png", stitched)
    if not ok:
        raise ValueError("Failed to encode stitched PDF as PNG")
    log("Stitched image size", f"{stitched.shape[1]}×{stitched.shape[0]} px")
    return buf.tobytes()


def to_png_bytes(raw_bytes: bytes) -> bytes:
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to re-encode image as PNG")
    return buf.tobytes()


def stage1_extract_markdown(image_bytes: bytes) -> str:
    log("STAGE 1 — Google Vision OCR")
    t0 = time.time()

    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_VISION_API_KEY not set")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
        }]
    }

    res = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
        json=payload,
        timeout=60
    )
    res.raise_for_status()

    result   = res.json()
    raw_text = result["responses"][0].get("fullTextAnnotation", {}).get("text", "")

    if not raw_text.strip():
        raise ValueError("Google Vision returned empty text")

    log("STAGE 1 Vision done", f"{round(time.time()-t0, 2)}s | {len(raw_text)} chars")

    prompt = f"""Convert this already-extracted text into clean Markdown.

CRITICAL RULES:
- Use ONLY the text provided below. Do NOT add, invent, or complete ANYTHING.
- Copy ALL text exactly as given — including [placeholders], [Your Company Name] etc.
- Square bracket text like [Your Name] must be copied EXACTLY — never replace with underscores.
- ONLY your job is formatting: headings (#), bold (**), bullets (-), tables (|col|col|)
- Do NOT complete sentences. Do NOT add words. Do NOT fix grammar.
- Output ONLY the Markdown. No explanation.

EXTRACTED TEXT:
{raw_text}"""

    payload2 = {
        "model": "openai/gpt-4o",
        "temperature": 0,
        "max_tokens": 4000,
        "messages": [
            {"role": "system", "content": "You are a text formatter only. Never invent content."},
            {"role": "user", "content": prompt},
        ],
    }
    res2 = requests.post(OPENROUTER_URL, headers=OCR_HEADERS, json=payload2, timeout=90)
    res2.raise_for_status()
    result2 = res2.json()["choices"][0]["message"]["content"]
    log("STAGE 1 formatting done", f"{round(time.time()-t0, 2)}s | {len(result2)} chars")
    return result2


def extract_json_safe(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log("⚠️ JSON parse failed, returning empty dict")
        return {}


def stage2_audit(raw_markdown: str) -> dict:
    log("STAGE 2 — Auditor")
    t0 = time.time()

    prompt = f"""You are a strict document auditor. Analyze the transcription below and return a JSON report.

TRANSCRIPTION:
```
{raw_markdown}
```

Check for ALL of the following:
1. Impossible dates (e.g. Feb 31, June 45, month > 12)
2. Broken or truncated words / sentences
3. Duplicate repeated paragraphs
4. Numbers that look obviously wrong (e.g. totals that don't add up)
5. [?] placeholders — note where they appear
6. Any text that looks invented rather than transcribed
7. Transposed digits in monetary amounts — flag ANY that might have swapped digits
8. Sentences that end unnaturally or seem completed/invented — flag as POTENTIAL_FABRICATION

CRITICAL — FABRICATION REMOVAL RULE:
If the LAST sentence or paragraph of the document ends abruptly, unnaturally,
or appears to complete a thought that was NOT fully visible in the image:
- Mark it as POTENTIAL_FABRICATION in issues_found
- In corrected_markdown: REMOVE that fabricated ending entirely.
  Cut the text at the last point you are CERTAIN was in the original image.
  End the document there — do NOT replace the fabrication with anything.
  It is better to have a document that ends mid-sentence with [DOCUMENT TRUNCATED]
  than to have invented content presented as fact.

Return ONLY this exact JSON (no extra text, no code fences):
{{
  "hallucination_risk": "low" | "medium" | "high",
  "issues_found": ["describe each issue, or empty array if none"],
  "illegible_fields": ["describe each [?] location, or empty array if none"],
  "corrections_made": ["describe each fix applied, or empty array if none"],
  "corrected_markdown": "<full corrected Markdown — remove any POTENTIAL_FABRICATION from the end>"
}}"""

    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": 4000,
        "messages": [
            {"role": "system", "content": "Return clean structured JSON only."},
            {"role": "user",   "content": prompt},
        ],
    }
    res    = requests.post(OPENROUTER_URL, headers=OCR_HEADERS, json=payload, timeout=90)
    res.raise_for_status()
    result = extract_json_safe(res.json()["choices"][0]["message"]["content"])
    risk   = result.get("hallucination_risk", "?").upper()
    log(f"STAGE 2 done {'🟢' if risk=='LOW' else '🟡' if risk=='MEDIUM' else '🔴'}",
        f"{round(time.time()-t0, 2)}s | risk={risk} | issues={len(result.get('issues_found', []))}")
    return result


def stage3_to_tiptap(markdown: str) -> dict:
    log("STAGE 3 — TipTap JSON")
    t0 = time.time()

    prompt = f"""Convert the following Markdown into a TipTap editor JSON document.

Rules:
- Root node: {{ "type": "doc", "content": [...] }}
- Supported node types: paragraph, heading (with level 1-6), bulletList, orderedList,
  listItem, blockquote, horizontalRule, table, tableRow, tableHeader, tableCell
- Supported marks: bold, italic, underline, strike
- Text nodes: {{ "type": "text", "text": "...", "marks": [...] }}
- Heading: {{ "type": "heading", "attrs": {{ "level": 1 }}, "content": [...] }}
- HorizontalRule (for page breaks): {{ "type": "horizontalRule" }}
- Output ONLY the JSON object. No explanation, no code fences.

MARKDOWN:
{markdown}"""

    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": 4000,
        "messages": [
            {"role": "system", "content": "Return valid TipTap JSON only."},
            {"role": "user",   "content": prompt},
        ],
    }
    res = requests.post(OPENROUTER_URL, headers=OCR_HEADERS, json=payload, timeout=90)
    res.raise_for_status()
    doc = extract_json_safe(res.json()["choices"][0]["message"]["content"])
    if doc.get("type") != "doc":
        doc = {"type": "doc", "content": doc.get("content", [])}
    log("STAGE 3 done", f"{round(time.time()-t0, 2)}s")
    return doc


def strip_truncated(markdown: str) -> str:
    marker = "[DOCUMENT TRUNCATED]"
    if marker in markdown:
        markdown = markdown[:markdown.index(marker)].rstrip()
        log("⚠️ TRUNCATION MARKER found — content cut at that point")
    return markdown


def parse_document(image_bytes: bytes) -> dict:
    try:
        log("START parse_document", f"bytes={len(image_bytes)}")
        t0 = time.time()

        raw_markdown = stage1_extract_markdown(image_bytes)
        if not raw_markdown.strip():
            raise ValueError("Stage 1 returned empty markdown")

        raw_markdown = strip_truncated(raw_markdown)

        audit             = stage2_audit(raw_markdown)
        risk              = audit.get("hallucination_risk", "low")
        verified_markdown = audit.get("corrected_markdown") or raw_markdown
        if not verified_markdown.strip():
            verified_markdown = raw_markdown

        verified_markdown = strip_truncated(verified_markdown)

        doc           = stage3_to_tiptap(verified_markdown)
        total_elapsed = round(time.time() - t0, 2)
        log("SUCCESS parse_document", f"total={total_elapsed}s | risk={risk}")

        doc["_audit"] = {
            "hallucination_risk": risk,
            "issues_found":       audit.get("issues_found", []),
            "corrections_made":   audit.get("corrections_made", []),
            "illegible_fields":   audit.get("illegible_fields", []),
            "pipeline_seconds":   total_elapsed,
            "engine_version":     ENGINE_VERSION,
        }
        return doc
    except Exception as e:
        log("❌ ERROR in parse_document", repr(e))
        traceback.print_exc()
        raise


def run_ocr_job(jobId: str):
    try:
        log("JOB START", jobId)
        job = load_job(jobId)
        if not job:
            raise RuntimeError("Job not found in Redis — possible cold start race condition")
        update_job(jobId, state="processing")

        file_path = job.get("filePath")
        if not file_path or not os.path.exists(file_path):
            raise RuntimeError(f"File not found: {file_path!r}")

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        image_bytes = pdf_to_image_bytes(raw_bytes) if is_pdf(raw_bytes) else to_png_bytes(raw_bytes)
        document    = parse_document(image_bytes)

        update_job(jobId, state="ready", contentJson=document)
        log("JOB DONE", jobId)
    except Exception as e:
        # CHANGE 3: store actual error so frontend can display it
        error_detail = repr(e)
        log("JOB ERROR", error_detail)
        traceback.print_exc()
        update_job(jobId, state="error", detail=error_detail)


# =============================================================
# ░░░░  SECTION 2 — DOCX EXPORT ENGINE  ░░░░░░░░░░░░░░░░░░░░░
# =============================================================

BODY_FONT = "Times New Roman"
BODY_SIZE = 12
H1_SIZE   = 22
H2_SIZE   = 13
H3_SIZE   = 11.5

H2_BORDER_COLOR = "2563EB"
H2_BORDER_SIZE  = 24

META_LABEL_FILL  = "EFF6FF"
META_HEADER_FILL = "F8FAFC"

DOC_LAYOUTS: dict = {
    "default":              {"shellVariant": "page",  "showLogo": False, "showSignature": False, "headerImageUrl": None, "footerImageUrl": None},
    "offer_modern_blue":    {"shellVariant": "page",  "showLogo": True,  "showSignature": True,  "headerImageUrl": "/graphics/offer/header-mod-blue.png",   "footerImageUrl": "/graphics/offer/footer-wave-blue.png"},
    "offer_green_wave":     {"shellVariant": "page",  "showLogo": True,  "showSignature": True,  "headerImageUrl": "/graphics/offer/header-green-wave.webp", "footerImageUrl": "/graphics/offer/footer-green-wave.webp"},
    "offer_minimal_plain":  {"shellVariant": "page",  "showLogo": True,  "showSignature": True,  "headerImageUrl": None, "footerImageUrl": None},
    "offer_classic_border": {"shellVariant": "page",  "showLogo": True,  "showSignature": True,  "headerImageUrl": None, "footerImageUrl": None},
    "noc_plain":            {"shellVariant": "page",  "showLogo": True,  "showSignature": True,  "headerImageUrl": None, "footerImageUrl": None},
    "rental_plain":         {"shellVariant": "page",  "showLogo": True,  "showSignature": True,  "headerImageUrl": None, "footerImageUrl": None},
    "plain_editor":         {"shellVariant": "plain", "showLogo": False, "showSignature": False, "headerImageUrl": None, "footerImageUrl": None},
}

SLUG_STYLE_OVERRIDES: dict = {
    "visa-expiration-letter":       "plain_editor",
    "website-proposal-standard":    "plain_editor",
    "mobile-app-proposal-standard": "plain_editor",
    "blog-article-standard":        "plain_editor",
    "rental-agreement-11-months":   "rental_plain",
    "offer-letter-standard":        "offer_modern_blue",
    "noc-employee-visa":            "noc_plain",
}


def get_layout(template_slug: Optional[str], design_key: Optional[str]) -> dict:
    s = (template_slug or "").lower()
    if s.startswith("leave-application-"):
        return DOC_LAYOUTS["plain_editor"]
    if design_key and design_key in DOC_LAYOUTS:
        return DOC_LAYOUTS[design_key]
    if template_slug:
        if template_slug in SLUG_STYLE_OVERRIDES:
            return DOC_LAYOUTS[SLUG_STYLE_OVERRIDES[template_slug]]
        if any(k in s for k in ("offer", "appointment", "joining")):
            return DOC_LAYOUTS["offer_modern_blue"]
        if "noc" in s:
            return DOC_LAYOUTS["noc_plain"]
        if any(k in s for k in ("rental", "lease")):
            return DOC_LAYOUTS["rental_plain"]
        if any(k in s for k in ("blog", "ai-blog", "content", "copywriter", "proposal")):
            return DOC_LAYOUTS["plain_editor"]
    return DOC_LAYOUTS["default"]


def configure_document_styles(document: Document):
    try:
        n = document.styles["Normal"]
        n.font.name = BODY_FONT
        n.font.size = Pt(BODY_SIZE)
    except KeyError:
        pass
    for name, size in (("Heading 1", H1_SIZE), ("Heading 2", H2_SIZE), ("Heading 3", H3_SIZE)):
        try:
            s = document.styles[name]
            s.font.name = BODY_FONT; s.font.size = Pt(size); s.font.bold = True
        except KeyError:
            pass


def apply_body_spacing(paragraph):
    fmt = paragraph.paragraph_format
    fmt.space_before      = Pt(0)
    fmt.space_after       = Pt(6)
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing      = 1.35


def set_page_margins(document: Document, top=1.0, bottom=1.0, left=1.0, right=1.0):
    s = document.sections[0]
    s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(top)


def add_left_border(paragraph, color=H2_BORDER_COLOR, size=H2_BORDER_SIZE):
    pPr  = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single"); left.set(qn("w:sz"), str(size))
    left.set(qn("w:space"), "12");   left.set(qn("w:color"), color)
    pBdr.append(left); pPr.append(pBdr)


def shade_cell(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def remove_cell_borders(cell):
    tcPr      = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{side}"); b.set(qn("w:val"), "none"); tcBorders.append(b)
    tcPr.append(tcBorders)


def add_bottom_border_to_cell(cell, color="334155", size=6):
    tcPr      = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    bottom    = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), str(size)); bottom.set(qn("w:color"), color)
    tcBorders.append(bottom); tcPr.append(tcBorders)


def fetch_image(url: Optional[str]) -> Optional[io.BytesIO]:
    if not url:
        return None
    try:
        if url.startswith("http"):
            r = requests.get(url, timeout=8); r.raise_for_status()
            return io.BytesIO(r.content)
        local = os.path.join(BASE_DIR, "public", url.lstrip("/"))
        if os.path.exists(local):
            with open(local, "rb") as f:
                return io.BytesIO(f.read())
    except Exception as e:
        print(f"⚠️  fetch_image failed for {url}: {e}")
    return None


def render_brand_header(document: Document, layout: dict, brand: Optional[dict], title: Optional[str]):
    header_img = fetch_image(layout.get("headerImageUrl"))
    if header_img:
        p = document.add_paragraph()
        p.add_run().add_picture(header_img, width=Inches(6.5))
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(6)

    if not brand:
        return

    tbl = document.add_table(rows=1, cols=3)
    tbl.autofit = True
    left_cell, center_cell, right_cell = tbl.rows[0].cells
    for cell in (left_cell, center_cell, right_cell):
        remove_cell_borders(cell)

    logo_img = fetch_image(brand.get("logoUrl")) if layout.get("showLogo") else None
    if logo_img:
        left_cell.text = ""
        left_cell.paragraphs[0].add_run().add_picture(logo_img, width=Inches(1.2))

    center_cell.text = ""
    nr = center_cell.paragraphs[0].add_run(brand.get("companyName", ""))
    nr.bold = True; nr.font.size = Pt(10); nr.font.name = BODY_FONT
    for line in (brand.get("addressLine1"), brand.get("addressLine2")):
        if line:
            p = center_cell.add_paragraph(line)
            for r in p.runs:
                r.font.size = Pt(8); r.font.name = BODY_FONT
                r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    right_cell.text = ""
    for val in (brand.get("phone"), brand.get("email")):
        if val:
            p = right_cell.add_paragraph(val)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for r in p.runs:
                r.font.size = Pt(8); r.font.name = BODY_FONT
                r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    if title:
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(title.upper())
        run.bold = True; run.font.size = Pt(9); run.font.name = BODY_FONT
        run.font.color.rgb = RGBColor(0x1D, 0x4E, 0xD8)
        pPr  = p._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        for side in ("top", "bottom"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "8"); b.set(qn("w:color"), "3B82F6")
            pBdr.append(b)
        pPr.append(pBdr)
        p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(8)

    document.add_paragraph()


def render_signatory_footer(document: Document, signatory: Optional[dict]):
    if not signatory:
        return
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_before = Pt(24); spacer.paragraph_format.space_after = Pt(4)

    lp = document.add_paragraph("Authorised Signatory")
    for r in lp.runs:
        r.bold = True; r.font.size = Pt(9); r.font.name = BODY_FONT
        r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    sig_img = fetch_image(signatory.get("signatureImageUrl"))
    if sig_img:
        p = document.add_paragraph()
        p.add_run().add_picture(sig_img, width=Inches(1.5))
        p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(2)
    else:
        blank = document.add_paragraph()
        blank.paragraph_format.space_before = Pt(16); blank.paragraph_format.space_after = Pt(2)

    for text, bold in ((signatory.get("fullName", ""), True), (signatory.get("designation", ""), False)):
        if text:
            p = document.add_paragraph(text)
            p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(1)
            for r in p.runs:
                r.bold = bold; r.font.size = Pt(BODY_SIZE if bold else 10); r.font.name = BODY_FONT
                if not bold: r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)


def render_footer_banner(document: Document, layout: dict):
    footer_img = fetch_image(layout.get("footerImageUrl"))
    if not footer_img:
        return
    p = document.add_paragraph()
    p.add_run().add_picture(footer_img, width=Inches(6.5))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(0)


def add_text_runs_from_tiptap(content_nodes: list, paragraph):
    if not content_nodes:
        return
    for node in content_nodes:
        ntype = node.get("type")

        if ntype == "text":
            text  = node.get("text", "")
            marks = node.get("marks", []) or []
            run   = paragraph.add_run(text)
            run.font.name = BODY_FONT; run.font.size = Pt(BODY_SIZE)
            for m in marks:
                mt = m.get("type")
                if mt == "bold":      run.bold        = True
                if mt == "italic":    run.italic      = True
                if mt == "underline": run.underline   = True
                if mt == "strike":    run.font.strike = True
                if mt == "textStyle":
                    color = m.get("attrs", {}).get("color", "")
                    if color and color.startswith("#") and len(color) == 7:
                        try:
                            run.font.color.rgb = RGBColor(
                                int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
                        except Exception:
                            pass
                if mt == "fontSize":
                    sz = m.get("attrs", {}).get("size")
                    if sz:
                        try:
                            run.font.size = Pt(float(re.sub(r"[^\d.]", "", str(sz))) * 0.75)
                        except Exception:
                            pass

        elif ntype == "formyxaField":
            attrs   = node.get("attrs", {}) or {}
            value   = (attrs.get("value") or "").strip()
            label   = (attrs.get("label") or "Field").strip()
            display = value if value else f"[{label}]"
            run = paragraph.add_run(display)
            run.font.name = BODY_FONT; run.font.size = Pt(BODY_SIZE)
            if attrs.get("bold"):  run.bold      = True
            if not value:          run.underline = True


def render_meta_table(node, document: Document):
    rows = node.get("content", [])
    if not rows: return
    num_rows = len(rows)
    num_cols = max(len(r.get("content", [])) for r in rows)
    table = document.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER; table.autofit = True

    for r_idx, row in enumerate(rows):
        for c_idx, cell_node in enumerate(row.get("content", [])):
            cell = table.rows[r_idx].cells[c_idx]; cell.text = ""
            for child in cell_node.get("content", []):
                if child.get("type") != "paragraph": continue
                p = cell.paragraphs[0]
                add_text_runs_from_tiptap(child.get("content", []) or [], p)
                p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(2)
            is_label = (c_idx % 2 == 0) or (cell_node.get("type") == "tableHeader")
            if is_label:
                shade_cell(cell, META_LABEL_FILL)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.bold = True; r.font.size = Pt(9); r.font.name = BODY_FONT; r.text = r.text.upper()
            else:
                shade_cell(cell, "FFFFFF")
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(BODY_SIZE); r.font.name = BODY_FONT
    document.add_paragraph()


def render_table_node(node, document: Document):
    rows = node.get("content", [])
    if not rows: return
    num_rows = len(rows)
    num_cols = max(len(r.get("content", [])) for r in rows)
    table = document.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER; table.autofit = True

    for r_idx, row in enumerate(rows):
        for c_idx, cell_node in enumerate(row.get("content", [])):
            cell = table.rows[r_idx].cells[c_idx]; cell.text = ""
            for child in cell_node.get("content", []):
                if child.get("type") != "paragraph": continue
                if (child.get("attrs") or {}).get("instructional"): continue
                content = child.get("content", []) or []
                if not any(c.get("type") == "text" and c.get("text", "").strip() for c in content): continue
                p = cell.paragraphs[0]
                add_text_runs_from_tiptap(content, p); apply_body_spacing(p)
            if cell_node.get("type") == "tableHeader":
                for p in cell.paragraphs:
                    for r in p.runs: r.bold = True
                shade_cell(cell, META_HEADER_FILL)


def render_signatures_block(node, document: Document, signatory: Optional[dict]):
    attrs       = node.get("attrs", {}) or {}
    left_title  = attrs.get("leftTitle",  "CLIENT")
    right_title = attrs.get("rightTitle", "SERVICE PROVIDER")

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_before = Pt(24); spacer.paragraph_format.space_after = Pt(6)

    table = document.add_table(rows=5, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER; table.autofit = True
    for row in table.rows:
        for cell in row.cells: remove_cell_borders(cell)

    def _set(cell, text, bold=False, size=BODY_SIZE):
        cell.text = ""
        p = cell.paragraphs[0]; run = p.add_run(text)
        run.font.name = BODY_FONT; run.font.size = Pt(size)
        if bold: run.bold = True
        p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(4)

    _set(table.rows[0].cells[0], left_title,  bold=True, size=10)
    _set(table.rows[0].cells[1], right_title, bold=True, size=10)

    for idx, cell in enumerate(table.rows[1].cells):
        sig_img = fetch_image(signatory.get("signatureImageUrl")) if (idx == 1 and signatory) else None
        if sig_img:
            p = cell.paragraphs[0]; p.paragraph_format.space_before = Pt(4)
            p.add_run().add_picture(sig_img, width=Inches(1.2))
            p.paragraph_format.space_after = Pt(2)
        else:
            cell.paragraphs[0].paragraph_format.space_before = Pt(24)
            cell.paragraphs[0].paragraph_format.space_after  = Pt(2)
            add_bottom_border_to_cell(cell)

    _set(table.rows[2].cells[0], "Signature", size=9)
    _set(table.rows[2].cells[1], "Signature", size=9)

    for cell in table.rows[3].cells:
        cell.text = ""; cell.paragraphs[0].paragraph_format.space_before = Pt(18)
        cell.paragraphs[0].paragraph_format.space_after = Pt(2); add_bottom_border_to_cell(cell)

    if signatory:
        rc = table.rows[3].cells[1]; rc.text = ""
        run = rc.paragraphs[0].add_run(f"{signatory.get('fullName', '')}  ({signatory.get('designation', '')})")
        run.font.size = Pt(9); run.font.name = BODY_FONT

    _set(table.rows[4].cells[0], "Name / Date", size=9)
    _set(table.rows[4].cells[1], "Name / Date", size=9)


def render_image_node(node, document: Document):
    src = (node.get("attrs") or {}).get("src")
    img = fetch_image(src)
    if not img: return
    p = document.add_paragraph()
    p.add_run().add_picture(img, width=Inches(4.5))
    apply_body_spacing(p)


def render_node(node, document: Document, signatory: Optional[dict] = None):
    ntype = node.get("type")

    if ntype == "heading":
        p = document.add_paragraph()
        add_text_runs_from_tiptap(node.get("content", []), p)
        for r in p.runs:
            r.bold = True
            r.font.name = BODY_FONT
            r.font.size = Pt(BODY_SIZE)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        apply_body_spacing(p)
        return

    if ntype == "paragraph":
        attrs   = node.get("attrs", {}) or {}
        if attrs.get("instructional"): return
        content = node.get("content", []) or []
        if not content:
            return
        has_text = any(
            c.get("type") == "text" and c.get("text", "").strip()
            for c in content
        )
        has_inline = any(c.get("type") in ("formyxaField", "hardBreak") for c in content)
        if not has_text and not has_inline:
            return
        p = document.add_paragraph()
        add_text_runs_from_tiptap(content, p)
        apply_body_spacing(p)
        align = (attrs.get("textAlign") or "").lower()
        if align == "center":    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "right":   p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        elif align == "justify": p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return

    if ntype == "bulletList":
        for li in node.get("content", []):
            if li.get("type") != "listItem": continue
            paras = [c for c in li.get("content", []) if c.get("type") == "paragraph"]
            for i, child in enumerate(paras):
                if (child.get("attrs") or {}).get("instructional"): continue
                content = child.get("content", []) or []
                if not content:
                    continue
                has_text = any(c.get("type") == "text" and c.get("text", "").strip() for c in content)
                has_field = any(c.get("type") == "formyxaField" for c in content)
                if not has_text and not has_field:
                    continue
                p = document.add_paragraph()
                if i == 0:
                    br = p.add_run("•  ")
                    br.font.name = BODY_FONT; br.font.size = Pt(BODY_SIZE)
                add_text_runs_from_tiptap(content, p)
                fmt = p.paragraph_format
                fmt.left_indent       = Pt(24)
                fmt.first_line_indent = Pt(-12) if i == 0 else Pt(0)
                fmt.space_before      = Pt(0)
                fmt.space_after       = Pt(3)
                fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
                fmt.line_spacing      = 1.35
        return

    if ntype == "orderedList":
        idx = 1
        for li in node.get("content", []):
            if li.get("type") != "listItem": continue
            paras = [c for c in li.get("content", []) if c.get("type") == "paragraph"]
            has_content = False
            for i, child in enumerate(paras):
                if (child.get("attrs") or {}).get("instructional"): continue
                content = child.get("content", []) or []
                if not content:
                    continue
                has_text = any(c.get("type") == "text" and c.get("text", "").strip() for c in content)
                has_field = any(c.get("type") == "formyxaField" for c in content)
                if not has_text and not has_field:
                    continue
                p = document.add_paragraph()
                if i == 0:
                    nr = p.add_run(f"{idx}.  ")
                    nr.font.name = BODY_FONT; nr.font.size = Pt(BODY_SIZE)
                add_text_runs_from_tiptap(content, p)
                fmt = p.paragraph_format
                fmt.left_indent       = Pt(24)
                fmt.first_line_indent = Pt(-12) if i == 0 else Pt(0)
                fmt.space_before      = Pt(0)
                fmt.space_after       = Pt(3)
                fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
                fmt.line_spacing      = 1.35
                has_content = True
            if has_content:
                idx += 1
        return

    if ntype == "table":
        cls = (node.get("attrs") or {}).get("class", "")
        (render_meta_table if cls == "meta-table" else render_table_node)(node, document)
        return

    if ntype == "signaturesBlock":
        render_signatures_block(node, document, signatory); return

    if ntype in ("image", "resizableImage"):
        render_image_node(node, document); return

    if ntype == "horizontalRule":
        p    = document.add_paragraph()
        pPr  = p._p.get_or_add_pPr(); pBdr = OxmlElement("w:pBdr")
        bot  = OxmlElement("w:bottom")
        bot.set(qn("w:val"), "single"); bot.set(qn("w:sz"), "6"); bot.set(qn("w:color"), "CBD5E1")
        pBdr.append(bot); pPr.append(pBdr)
        p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(12)
        return

    if ntype == "pageBreak":
        p   = document.add_paragraph()
        pPr = p._p.get_or_add_pPr()
        pb  = OxmlElement("w:pageBreakBefore"); pb.set(qn("w:val"), "true"); pPr.append(pb)
        return


def tiptap_doc_to_docx(
    tiptap_doc:    Optional[dict],
    template_slug: Optional[str]  = None,
    design_key:    Optional[str]  = None,
    brand:         Optional[dict] = None,
    signatory:     Optional[dict] = None,
    file_name:     str            = "document",
) -> Document:

    document = Document(BASE_TEMPLATE) if os.path.exists(BASE_TEMPLATE) else Document()
    configure_document_styles(document)
    set_page_margins(document)
    layout = get_layout(template_slug, design_key)

    if layout.get("showLogo") or layout.get("headerImageUrl"):
        render_brand_header(document, layout, brand, file_name)

    if tiptap_doc and tiptap_doc.get("type") == "doc":
        for node in tiptap_doc.get("content", []):
            render_node(node, document, signatory=signatory)

    if layout.get("showSignature") and signatory:
        render_signatory_footer(document, signatory)

    if layout.get("footerImageUrl"):
        render_footer_banner(document, layout)

    return document


def sanitize_filename(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s\-.]", "_", name)
    return name.strip() or "document"


# =============================================================
# ░░░░  SECTION 3 — ALL ROUTES  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# =============================================================

class ProcessRequest(BaseModel):
    jobId: str

@app.post("/api/handwritten/process")
async def start_handwritten_process(payload: ProcessRequest, background_tasks: BackgroundTasks):
    log("Starting background OCR job", payload.jobId)
    update_job(payload.jobId, state="queued")
    background_tasks.add_task(run_ocr_job, payload.jobId)
    return {"started": True}


@app.post("/api/job-register")
async def register_job(payload: dict):
    jobId = payload["jobId"]
    update_job(jobId, filePath=payload["filePath"], source=payload.get("source", "scanned"),
               strict=payload.get("strict", True), state="uploaded")
    return {"ok": True}


@app.get("/api/job-status")
async def job_status(jobId: str):
    job = load_job(jobId)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/job-complete-free")
async def complete_free_job(payload: dict):
    update_job(payload["jobId"], state="free-ready", source="digital-pdf")
    return {"ok": True}


@app.post("/api/detect-pdf-type")
async def detect_pdf_type_route(file: UploadFile = File(...)):
    data = await file.read()
    doc  = fitz.open(stream=data, filetype="pdf")
    for i in range(min(len(doc), 3)):
        if doc.load_page(i).get_text().strip():
            return {"type": "digital"}
    return {"type": "scanned"}


class ExportRequest(BaseModel):
    filePath: str

@app.post("/api/export-digital-docx")
async def export_digital_docx(payload: ExportRequest):
    if not os.path.exists(payload.filePath):
        raise HTTPException(status_code=400, detail="FILE_NOT_FOUND")
    with open(payload.filePath, "rb") as f:
        pdf_bytes = f.read()
    pdf      = fitz.open(stream=pdf_bytes, filetype="pdf")
    word_doc = Document()
    s = word_doc.sections[0]
    s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(1)
    for page in pdf:
        raw_text = page.get_text().strip()
        if not raw_text: continue
        for block in [b.strip() for b in raw_text.split("\n\n") if b.strip()]:
            p = word_doc.add_paragraph(block)
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_after  = Pt(12)
            p.paragraph_format.space_before = Pt(0)
            for run in p.runs:
                run.font.name = "Times New Roman"; run.font.size = Pt(12)
    buf = io.BytesIO()
    word_doc.save(buf); buf.seek(0)
    return StreamingResponse(buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=Converted_Document.docx"})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", f"{datetime.now().timestamp()}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"filePath": file_path}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="UPLOAD_FAILED")


@app.post("/api/parse-document")
async def parse_document_route(
    file:   UploadFile = File(...),
    strict: bool       = Form(True),
    source: str        = Form("scanned"),
):
    log("API HIT /api/parse-document")
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="EMPTY_FILE")
        image_bytes = pdf_to_image_bytes(raw_bytes) if is_pdf(raw_bytes) else to_png_bytes(raw_bytes)
        document    = parse_document(image_bytes)
        return {"success": True, "engine_version": ENGINE_VERSION, "document": document}
    except HTTPException:
        raise
    except Exception as e:
        log("❌ ROUTE ERROR", repr(e)); traceback.print_exc()
        raise HTTPException(status_code=500, detail="PROCESSING_FAILED")


class GenerateDocxRequest(BaseModel):
    contentJson:  Any           = None
    fileName:     Optional[str] = Field(default="document")
    templateSlug: Optional[str] = None
    designKey:    Optional[str] = None
    brand:        Optional[Any] = None
    signatory:    Optional[Any] = None
    baseTemplate: Optional[str] = None


@app.post("/generate-docx")
async def generate_docx_route(payload: GenerateDocxRequest):
    try:
        log("GENERATE DOCX", f"slug={payload.templateSlug} design={payload.designKey} file={payload.fileName}")

        document = tiptap_doc_to_docx(
            tiptap_doc    = payload.contentJson,
            template_slug = payload.templateSlug,
            design_key    = payload.designKey,
            brand         = payload.brand,
            signatory     = payload.signatory,
            file_name     = payload.fileName or "document",
        )

        buf = io.BytesIO()
        document.save(buf); buf.seek(0)

        safe_name = sanitize_filename(payload.fileName or "document")
        if not safe_name.lower().endswith(".docx"):
            safe_name += ".docx"

        return Response(
            content=buf.read(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)