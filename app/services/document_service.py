"""Validate uploads and extract text with local libraries only (no AI API involved)."""
import csv
import io
import json
import logging
import re
import statistics
import zipfile
from dataclasses import dataclass, field

from app.config import Settings
from app.utils.errors import AppError

log = logging.getLogger("searchly.documents")

ALLOWED = {
    ".pdf": "pdf", ".txt": "text", ".md": "markdown", ".markdown": "markdown", ".docx": "docx", ".csv": "csv",
    ".xlsx": "xlsx", ".json": "json", ".png": "image", ".jpg": "image", ".jpeg": "image",
}
CHUNK_CHARS = 1400
MAX_CHUNKS = 400
MAX_PDF_PAGES = 300
MAX_ROWS = 20000
MAX_UNZIPPED = 200 * 1024 * 1024


@dataclass
class Chunk:
    text: str
    location: str
    priority: bool = False


@dataclass
class ProcessedDocument:
    filename: str
    file_type: str
    chunks: list[Chunk]
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def sanitize_filename(name: str) -> str:
    base = (name or "").replace("\\", "/").split("/")[-1]  # strips any path (path traversal)
    base = re.sub(r"[^\w.\- ]+", "_", base, flags=re.UNICODE).strip(" .")
    return base[:120] or "upload"


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]


def split_text(text: str, location_prefix: str, size: int = CHUNK_CHARS) -> list[Chunk]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        while len(p) > size:  # very long paragraph: hard split
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(p[:size])
            p = p[size:]
        if len(buf) + len(p) + 2 > size and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [Chunk(c, f"{location_prefix}, part {i + 1}" if len(chunks) > 1 and location_prefix else location_prefix)
            for i, c in enumerate(chunks)]


class DocumentService:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ---------- validation ----------
    def validate(self, filename: str, data: bytes) -> tuple[str, str]:
        clean = sanitize_filename(filename)
        ext = "." + clean.rsplit(".", 1)[-1].lower() if "." in clean else ""
        if ext not in ALLOWED:
            raise AppError(415, "unsupported_file_type",
                           "This file type is not supported. Use PDF, TXT, DOCX, CSV, XLSX, JSON, Markdown, PNG or JPG.")
        if len(data) == 0:
            raise AppError(422, "empty_document", "The file is empty.")
        if len(data) > self.settings.max_upload_bytes:
            raise AppError(413, "file_too_large", f"The file is larger than the {self.settings.max_upload_mb} MB limit.")
        kind = ALLOWED[ext]
        bad_sig = AppError(415, "invalid_file_signature", "The file contents do not match its extension, so it was rejected.")
        if kind == "pdf" and not data.lstrip()[:5].startswith(b"%PDF"):
            raise bad_sig
        if kind == "image":
            png, jpg = data.startswith(b"\x89PNG\r\n\x1a\n"), data.startswith(b"\xff\xd8\xff")
            if not ((ext == ".png" and png) or (ext in (".jpg", ".jpeg") and jpg)):
                raise bad_sig
        if kind in ("docx", "xlsx"):
            if not data.startswith(b"PK\x03\x04"):
                raise bad_sig
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    names = set(z.namelist())
                    if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED:
                        raise AppError(413, "file_too_large", "The file expands to an unsafe size and was rejected.")
            except zipfile.BadZipFile:
                raise bad_sig
            needed = "word/document.xml" if kind == "docx" else "xl/workbook.xml"
            if needed not in names:
                raise bad_sig
        if kind in ("text", "markdown", "csv", "json") and b"\x00" in data[:8192] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise bad_sig
        return clean, kind

    # ---------- extraction ----------
    def process(self, filename: str, data: bytes) -> ProcessedDocument:
        clean, kind = self.validate(filename, data)
        try:
            handler = getattr(self, f"_read_{kind}")
            doc = handler(clean, data)
        except AppError:
            raise
        except Exception as exc:
            log.warning("could not read %s file: %s", kind, type(exc).__name__)
            raise AppError(422, "unreadable_document", "The file could not be read. It may be damaged or password-protected.")
        doc.chunks = [c for c in doc.chunks if c.text.strip()]
        if not doc.chunks or sum(len(c.text) for c in doc.chunks) < 3:
            raise AppError(422, "empty_document", "No readable text was found in this file.")
        if len(doc.chunks) > MAX_CHUNKS:
            doc.chunks = doc.chunks[:MAX_CHUNKS]
            doc.warnings.append("document_truncated")
        doc.stats["chunks"] = len(doc.chunks)
        doc.stats["characters"] = sum(len(c.text) for c in doc.chunks)
        return doc

    @staticmethod
    def _decode(data: bytes) -> str:
        encodings = ["utf-8-sig"]
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 only when it has a byte-order mark
            encodings.insert(0, "utf-16")
        for enc in encodings:
            try:
                return data.decode(enc)
            except UnicodeError:
                continue
        raise AppError(422, "unsupported_encoding", "The text file is not UTF-8. Please re-save it as UTF-8 and upload again.")

    def _read_text(self, name, data):
        return ProcessedDocument(name, "text", split_text(self._decode(data), "text"))

    def _read_markdown(self, name, data):
        text = self._decode(data)
        parts = re.split(r"(?m)^(?=#{1,3} )", text)
        chunks = []
        for part in parts:
            m = re.match(r"#{1,3} (.+)", part)
            label = f"section '{m.group(1).strip()[:60]}'" if m else "introduction"
            chunks.extend(split_text(part, label))
        return ProcessedDocument(name, "markdown", chunks)

    def _read_json(self, name, data):
        try:
            obj = json.loads(self._decode(data))
        except json.JSONDecodeError:
            raise AppError(422, "invalid_json", "This JSON file is not valid JSON.")
        if isinstance(obj, list) and obj:
            stats = {"items": len(obj)}
            chunks = []
            for i in range(0, len(obj), 20):
                chunks.append(Chunk(json.dumps(obj[i:i + 20], ensure_ascii=False, indent=1), f"items {i + 1}-{min(i + 20, len(obj))}"))
            result = []
            for c in chunks:
                result.extend(split_text(c.text, c.location))
            return ProcessedDocument(name, "json", result, stats)
        return ProcessedDocument(name, "json", split_text(json.dumps(obj, ensure_ascii=False, indent=1), "json"))

    def _read_pdf(self, name, data):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    raise AppError(422, "encrypted_pdf", "This PDF is password-protected.")
            except AppError:
                raise
            except Exception:
                raise AppError(422, "encrypted_pdf", "This PDF is password-protected.")
        total = len(reader.pages)
        chunks, warnings = [], []
        for i, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
            chunks.extend(split_text(page.extract_text() or "", f"page {i}"))
        if total > MAX_PDF_PAGES:
            warnings.append("document_truncated")
        if not chunks:
            warnings.append("pdf_has_no_text_layer")
        return ProcessedDocument(name, "pdf", chunks, {"pages": total}, warnings)

    def _read_docx(self, name, data):
        from docx import Document
        d = Document(io.BytesIO(data))
        chunks, section, buf = [], "start", []

        def flush():
            if buf:
                chunks.extend(split_text("\n\n".join(buf), f"section '{section}'"))
                buf.clear()

        for p in d.paragraphs:
            if p.style is not None and p.style.name.lower().startswith("heading") and p.text.strip():
                flush()
                section = p.text.strip()[:60]
            if p.text.strip():
                buf.append(p.text)
        flush()
        for ti, table in enumerate(d.tables, start=1):
            rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
            chunks.extend(split_text("\n".join(rows), f"table {ti}"))
        return ProcessedDocument(name, "docx", chunks, {"paragraphs": len(d.paragraphs), "tables": len(d.tables)})

    def _table_chunks(self, label: str, header: list[str], rows: list[list[str]], first_row_no: int) -> list[Chunk]:
        """Summary chunk (columns + numeric stats) followed by row-range chunks."""
        lines = [f"{label}: {len(rows)} data rows, {len(header)} columns.", "Columns: " + ", ".join(header)]
        for ci, col in enumerate(header):
            vals = []
            for r in rows:
                if ci < len(r):
                    try:
                        vals.append(float(str(r[ci]).replace(",", "")))
                    except ValueError:
                        pass
            if len(vals) >= max(3, len(rows) * 0.6):
                lines.append(f"- {col}: min {min(vals):g}, max {max(vals):g}, mean {statistics.fmean(vals):.4g}, sum {sum(vals):.6g} ({len(vals)} numeric values)")
        chunks = [Chunk("\n".join(lines), f"{label} summary", priority=True)]
        head, step, i = " | ".join(header), 25, 0
        while i < len(rows):
            part = rows[i:i + step]
            body = "\n".join(" | ".join(str(c) for c in r) for r in part)
            chunks.append(Chunk(f"{head}\n{body}", f"{label}, rows {first_row_no + i}-{first_row_no + i + len(part) - 1}"))
            i += step
        return chunks

    def _read_csv(self, name, data):
        text = self._decode(data)
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = []
        for i, row in enumerate(csv.reader(io.StringIO(text), dialect)):
            if i > MAX_ROWS:
                break
            if any(c.strip() for c in row):
                rows.append(row)
        if len(rows) < 1:
            raise AppError(422, "empty_document", "The CSV file has no rows.")
        header, body = rows[0], rows[1:]
        stats = {"rows": len(body), "columns": header}
        return ProcessedDocument(name, "csv", self._table_chunks("CSV", header, body, 2), stats)

    def _read_xlsx(self, name, data):
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        chunks, sheets = [], {}
        for ws in wb.worksheets:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i > MAX_ROWS:
                    break
                cells = ["" if v is None else str(v) for v in row]
                if any(c.strip() for c in cells):
                    rows.append(cells)
            if not rows:
                continue
            header, body = rows[0], rows[1:]
            sheets[ws.title] = {"rows": len(body), "columns": header}
            chunks.extend(self._table_chunks(f"sheet '{ws.title}'", header, body, 2))
        wb.close()
        return ProcessedDocument(name, "xlsx", chunks, {"sheets": sheets})

    def _read_image(self, name, data):
        try:
            import pytesseract
            from PIL import Image
            pytesseract.get_tesseract_version()
        except Exception:
            raise AppError(503, "ocr_unavailable",
                           "Reading text from images needs Tesseract OCR installed on the server. See the README (OCR section), or upload a PDF/text file instead.")
        Image.MAX_IMAGE_PIXELS = 40_000_000
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:
            raise AppError(422, "unreadable_document", "The image could not be opened.")
        try:
            text = pytesseract.image_to_string(img, lang=self.settings.ocr_languages)
        except Exception:
            raise AppError(422, "ocr_failed", "OCR failed. If you use non-English text, install the matching Tesseract language data and set OCR_LANGUAGES.")
        if not text.strip():
            raise AppError(422, "empty_document", "No text could be recognised in this image.")
        return ProcessedDocument(name, "image", split_text(text, "image text (OCR)"), {"width": img.width, "height": img.height},
                                 ["ocr_text_may_contain_errors"])


def retrieve(chunks: list[Chunk], question: str, k: int = 8) -> list[Chunk]:
    """Pick the chunks most relevant to the question (keyword scoring, works for any language)."""
    if len(chunks) <= k:
        return list(chunks)
    q = set(_tokens(question))
    df: dict[str, int] = {}
    toks = [set(_tokens(c.text)) for c in chunks]
    for t in toks:
        for w in t & q:
            df[w] = df.get(w, 0) + 1
    scored = []
    for i, (c, t) in enumerate(zip(chunks, toks)):
        s = sum(1.0 / (1 + df[w]) + 0.2 for w in t & q)
        s += 0.5 if c.priority else 0
        scored.append((s, i))
    scored.sort(reverse=True)
    picked = {i for s, i in scored[:k] if s > 0}
    if len(picked) < k // 2:  # generic question such as "summarize": sample across the whole document
        step = max(1, len(chunks) // (k - len(picked)))
        for i in range(0, len(chunks), step):
            picked.add(i)
            if len(picked) >= k:
                break
    return [chunks[i] for i in sorted(picked)][:k]
