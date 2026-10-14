"""Rebuild the workflow SVGs; add --png for README PNGs (requires Pillow).

Run from any directory: python docs/diagrams/generate_workflows.py --png
Diagrams describe repository code; no environment files or credentials are read.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent
INK = "#142b43"
MUTED = "#496079"
LINE = "#72879b"
PALETTES = {
    "neutral": ("#ffffff", "#d6e1eb", INK),
    "java": ("#eaf7f3", "#a2d7c7", "#11634d"),
    "python": ("#edf3ff", "#b1c7ed", "#254f99"),
    "shared": ("#f2efff", "#c9bfec", "#63459b"),
    "success": ("#eaf7ee", "#a9d7b5", "#25653a"),
    "error": ("#fff1ee", "#efb9ac", "#a04432"),
    "note": ("#fff8e7", "#e9d59e", "#7b5c17"),
}


class Diagram:
    def __init__(self, slug, title, subtitle, height, png=False):
        self.slug, self.title, self.subtitle = slug, title, subtitle
        self.width, self.height = 1440, height
        self.elements = []
        self.draw = None
        if png:
            from PIL import Image, ImageDraw, ImageFont

            self.im = Image.new("RGB", (self.width, self.height), "#f6f8fc")
            self.draw = ImageDraw.Draw(self.im)
            font_dir = Path("C:/Windows/Fonts")
            self.fonts = {
                (size, bold): ImageFont.truetype(str(font_dir / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)
                for size in (16, 18, 20, 22, 34)
                for bold in (False, True)
            }
        self.rect(0, 0, 1440, 120, "#142b43", "#142b43", radius=0)
        self.text(48, 24, title, 34, "#ffffff", bold=True)
        self.text(48, 76, subtitle, 18, "#cfdeed")

    def rect(self, x, y, w, h, fill, stroke, radius=16):
        self.elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        if self.draw:
            self.draw.rounded_rectangle((x, y, x + w, y + h), radius, fill=fill, outline=stroke, width=2)

    def text(self, x, y, value, size=18, color=MUTED, bold=False):
        self.elements.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" dominant-baseline="text-before-edge">{escape(value)}</text>')
        if self.draw:
            self.draw.text((x, y), value, fill=color, font=self.fonts[size, bold], anchor="lt")

    def box(self, x, y, w, h, title, lines=(), tone="neutral"):
        fill, border, accent = PALETTES[tone]
        self.rect(x, y, w, h, fill, border)
        self.rect(x + 18, y + 22, 5, h - 44, accent, accent, radius=2)
        self.text(x + 36, y + 20, title, 22, accent, bold=True)
        for i, line in enumerate(lines):
            self.text(x + 36, y + 58 + i * 27, line)
        assert 58 + len(lines) * 27 <= h + 4, (self.slug, title, "text overflows height")

    def arrow(self, points, label=None, label_at=None, color=LINE):
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        x, y = points[-1]
        px, py = points[-2]
        if x == px:
            s = 1 if y > py else -1
            head = [(x, y), (x - 7, y - s * 12), (x + 7, y - s * 12)]
        else:
            s = 1 if x > px else -1
            head = [(x, y), (x - s * 12, y - 7), (x - s * 12, y + 7)]
        self.elements.append(f'<polygon points="{" ".join(f"{a},{b}" for a, b in head)}" fill="{color}"/>')
        if self.draw:
            self.draw.line(points, fill=color, width=3)
            self.draw.polygon(head, fill=color)
        if label:
            lx, ly = label_at
            self.text(lx, ly, label, 16, color, bold=True)

    def save(self):
        self.text(48, self.height - 38, "GATHER  /  Implementation workflows  /  Java + Python are primary backend options", 16)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{self.height}" viewBox="0 0 1440 {self.height}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{escape(self.title)}</title><desc id="desc">{escape(self.subtitle)}</desc>'
            '<g font-family="Segoe UI, Arial, sans-serif">'
            f'<rect width="1440" height="{self.height}" fill="#f6f8fc"/>'
            + "".join(self.elements) + "</g></svg>"
        )
        (OUT / f"{self.slug}.svg").write_text(svg, encoding="utf-8")
        if self.draw:
            self.im.save(OUT / f"{self.slug}.png", optimize=True)
        print(f"Generated {self.slug}")


def architecture(png):
    d = Diagram("01-system-architecture", "01  System architecture", "Choose either primary backend. Both expose collection, document, chat, session, and audit APIs.", 1260, png)
    d.arrow([(720, 300), (720, 335), (375, 335), (375, 375)])
    d.arrow([(720, 335), (1065, 335), (1065, 375)])
    d.arrow([(375, 645), (375, 710)])
    d.arrow([(1065, 645), (1065, 710)])
    d.arrow([(60, 510), (28, 510), (28, 1000), (60, 1000)])
    d.arrow([(1380, 510), (1412, 510), (1412, 1000), (1380, 1000)])
    d.box(320, 155, 800, 145, "Browser workspace / React UI", ["Vite: localhost:5173/react/; API proxy targets localhost:8080.", "Java also serves the built-in UI at / and packaged React at /react/.", "Select a backend on port 8080, or update the proxy for another port."], "shared")
    d.box(60, 375, 630, 270, "JAVA  |  Primary backend", ["Spring Boot API + Spring Security sessions", "RagController routes requests to modular services.", "Ingestion: PDFBox for PDF; Tika for DOCX/TXT.", "Retrieval: vector search; full-collection summaries.", "Answers: Spring AI / OpenAI; extractive fallback.", "Bounded async ingestion executor + audit records."], "java")
    d.box(750, 375, 630, 270, "PYTHON  |  Primary backend", ["FastAPI API + in-memory sessions", "app/api.py coordinates ingestion, retrieval, and chat.", "Ingestion: Docling/OCR or native format parsers.", "Retrieval: cached TF-IDF + optional dense rankings.", "Answers: OpenAI orchestration; extractive fallback.", "Async tasks + six ingestion slots + audit records."], "python")
    d.box(60, 710, 630, 180, "Java storage options", ["Memory: collections and chunks clear on restart.", "PostgreSQL + pgvector: persistent metadata/chunks.", "Cosine search + HNSW; queries scoped by collection.", "Docker Compose deploys Java + React + PostgreSQL."], "java")
    d.box(750, 710, 630, 180, "Python storage", ["MemoryStore: collections, chunks, vectors, and audit.", "Per-collection TF-IDF index cache for retrieval.", "Application state clears when the process stops.", "Run with uvicorn; separate from Compose deployment."], "python")
    d.box(60, 950, 1320, 145, "Optional external providers  |  Calls originate from the selected backend", ["Both: OpenAI embeddings, answer generation, or hosted File Search with a collection_id filter.", "Python also supports local Hugging Face Transformer embeddings; models download on first use.", "Hosted retrieval uploads originals to OpenAI; local retrieval parses, chunks, embeds, and indexes locally."], "shared")
    d.box(60, 1130, 1320, 76, "Collections scope retrieval; authenticated accounts share instance-wide collection access.", tone="note")
    d.save()


def ingestion(png):
    d = Diagram("02-document-ingestion", "02  Document ingestion lifecycle", "Shared upload lifecycle, with explicit Java and Python processing paths and retry behavior.", 1800, png)
    d.arrow([(700, 290), (700, 345)])
    d.arrow([(700, 485), (700, 540)])
    d.arrow([(700, 680), (700, 735)])
    d.arrow([(980, 410), (1040, 410)])
    d.arrow([(980, 610), (1040, 610)])
    d.arrow([(700, 880), (700, 920), (270, 920), (270, 970)], "Local / Java", (90, 930))
    d.arrow([(700, 920), (720, 920), (720, 970)], "Local / Python", (600, 930))
    d.arrow([(700, 920), (1170, 920), (1170, 970)], "Hosted / either", (1060, 930))
    d.arrow([(270, 1180), (270, 1235)])
    d.arrow([(720, 1180), (720, 1235)])
    d.arrow([(270, 1415), (270, 1470), (700, 1470), (700, 1510)])
    d.arrow([(720, 1415), (720, 1510)])
    d.arrow([(1170, 1180), (1170, 1470), (720, 1470)])
    d.box(420, 155, 560, 135, "1  Upload to the selected collection", ["POST /api/collections/{id}/documents", "Multipart file + session cookie + CSRF header.", "ADMIN or EDITOR role required."], "shared")
    d.box(420, 345, 560, 140, "2  Validate the upload", ["Collection exists; filename reduced to basename.", "Non-empty PDF, DOCX, or TXT; default limit 10 MB.", "Check format signature / detected content type."])
    d.box(1040, 345, 340, 140, "Rejected request", ["Invalid type, size, or collection.", "Return an API error.", "No ingestion is scheduled."], "error")
    d.box(420, 540, 560, 140, "3  Deduplicate with SHA-256", ["Lookup checksum within the selected collection.", "Existing READY / PROCESSING: return HTTP 200.", "Existing FAILED: retry the same document."], "shared")
    d.box(1040, 540, 340, 140, "Already present", ["Return existing document.", "Reuse its ID and status.", "Avoid duplicate indexing."], "success")
    d.box(420, 735, 560, 145, "4  Reserve capacity and queue processing", ["Mark PROCESSING; HTTP 202 when accepted.", "Java: bounded executor. Python: six task slots.", "Full capacity: return an error; retry shortly."], "shared")
    d.box(60, 970, 420, 210, "5A  Java local parsing", ["PDF: PDFBox, preserving page numbers.", "DOCX/TXT: Apache Tika extraction.", "Normalize text; overlapping chunks.", "Defaults: 1,200 chars / 180 overlap.", "Reject empty text or >1,800 chunks."], "java")
    d.box(510, 970, 420, 210, "5B  Python local parsing", ["Docling: layout, tables, optional OCR.", "Auto mode can use native fallback.", "Native: pypdf / python-docx / TXT.", "Build chunks + source locations.", "Reject empty text or >1,800 chunks."], "python")
    d.box(960, 970, 420, 210, "5C  Hosted File Search", ["Upload original file to OpenAI.", "Attach to configured vector store.", "Set collection_id + document_id.", "Poll until indexing completes.", "Persist the remote file identifier."], "shared")
    d.box(60, 1235, 420, 180, "6A  Java embeddings + storage", ["Feature hashing or OpenAI vectors.", "Embed chunks; preserve PDF pages.", "Store in memory or PostgreSQL.", "Complete the document index."], "java")
    d.box(510, 1235, 420, 180, "6B  Python embeddings + storage", ["Hashing / Transformers / OpenAI.", "Embed chunks into MemoryStore.", "Keep text + location metadata.", "TF-IDF cache refreshes on retrieval."], "python")
    d.box(420, 1510, 560, 135, "7  READY and available for questions", ["Record ingestion-completed audit event.", "React polls document status every 2.5 seconds.", "Only READY documents supply evidence."], "success")
    d.box(60, 1680, 1320, 76, "Worker failure -> FAILED + error + audit; re-upload the same file to retry processing.", tone="error")
    d.save()


def questions(png):
    d = Diagram("03-question-answering", "03  Collection question answering", "The API chooses local retrieval or hosted File Search. Provider failure and evidence checks are shown separately.", 1710, png)
    d.arrow([(720, 300), (720, 350), (375, 350), (375, 395)], "Local retrieval", (230, 360))
    d.arrow([(720, 350), (1110, 350), (1110, 395)], "Hosted retrieval", (1010, 360))
    d.arrow([(375, 625), (375, 690)])
    d.arrow([(690, 760), (755, 760)])
    d.arrow([(375, 830), (375, 895)])
    d.arrow([(1110, 625), (1410, 625), (1410, 1000), (1380, 1000)])
    d.arrow([(375, 1125), (375, 1190)])
    d.arrow([(1110, 1105), (1250, 1105), (1250, 1520), (1120, 1520)])
    d.arrow([(375, 1330), (375, 1410), (720, 1410), (720, 1450)])
    d.box(320, 155, 800, 145, "1  Submit and validate a question", ["POST /api/collections/{id}/chat; authenticated session + CSRF.", "Require an existing collection and non-empty question (up to 2,000 chars).", "Python resolves follow-ups from the last eight messages; Java uses the question."], "shared")
    d.box(60, 395, 630, 230, "2A  Retrieve local evidence", ["Java: embed query -> collection-scoped vector search.", "Java summaries: candidates from the full collection.", "Python: spelling/context handling -> hybrid retrieval.", "Python date questions can use direct fact extraction.", "Use READY documents; default top-k 10 (range 3-20)."], "shared")
    d.box(840, 395, 540, 230, "2B  OpenAI hosted File Search", ["Call Responses with the file_search tool.", "Filter results by collection_id.", "Map remote file IDs to local READY documents.", "Require both answer text and usable citations.", "Missing configuration / service failure -> API error."], "shared")
    d.box(60, 690, 630, 140, "3  Evidence available?", ["No matches -> explicit no-evidence response.", "If ingestion is active, ask the user to retry shortly.", "With matches, build labeled source passages."])
    d.box(755, 690, 625, 140, "No evidence", ["Return message + grounded: false.", "No local source matches -> sources: [].", "Record NO_EVIDENCE / PROCESSING audit outcome."], "error")
    d.box(60, 895, 630, 230, "4A  Generate from retrieved passages", ["Java: Spring AI when selected/available; OpenAI path.", "Python: OpenAI agent orchestration when available.", "Missing credentials / provider failure: extractive fallback.", "Prompt: document text is evidence, never instructions.", "Python summaries can use full-collection fallback context."], "shared")
    d.box(840, 895, 540, 210, "4B  Hosted answer + citations", ["Return provider answer with matched file excerpts.", "No answer or citations -> grounded: false.", "Hosted citations do not include local PDF pages.", "Hosted failures return errors; no local-search fallback."], "shared")
    d.box(60, 1190, 630, 140, "5  Build local citations", ["Document ID/name, chunk index, excerpt, PDF page.", "Python also supplies sourceLocation when available.", "Python rejects unsupported extractive answers."])
    d.box(320, 1450, 800, 145, "6  Return response and render sources", ["ChatResponse contains answer, sources, and grounded fields.", "Record question-answer audit outcome; browser shows answer and source cards.", "Java currently sets grounded: true whenever local retrieval found matches."], "success")
    d.save()


def python_retrieval(png):
    d = Diagram("04-python-hybrid-retrieval", "04  Python retrieval and answer routing", "Python-specific details: query preparation, date extraction, lexical/dense fusion, and evidence recovery.", 1800, png)
    d.arrow([(720, 300), (720, 350)])
    d.arrow([(1000, 420), (1050, 420)])
    d.arrow([(720, 490), (720, 545)])
    d.arrow([(1000, 615), (1050, 615)])
    d.arrow([(720, 685), (720, 730), (270, 730), (270, 785)])
    d.arrow([(720, 730), (720, 785)])
    d.arrow([(720, 730), (1170, 730), (1170, 785)])
    for x in (270, 720, 1170):
        d.arrow([(x, 995), (x, 1040), (720, 1040), (720, 1090)])
    d.arrow([(720, 1235), (720, 1290)])
    d.arrow([(720, 1435), (720, 1490)])
    d.box(320, 155, 800, 145, "1  Build collection context and normalize the query", ["Use READY chunks, section outlines, and source locations from this collection.", "Correct query spelling; contextualize follow-ups with the last eight messages.", "Detect overview intent and extract date/deadline facts where applicable."], "python")
    d.box(440, 350, 560, 140, "2  Date or deadline facts found?", ["Yes: format facts + citations and return immediately.", "No: continue to configured retrieval provider.", "Overview intent can expand retrieval to 20 results."], "python")
    d.box(1050, 350, 330, 140, "Direct fact answer", ["Cite the source chunks.", "Set grounded: true.", "Record GROUNDED audit."], "success")
    d.box(440, 545, 560, 140, "3  Select retrieval provider", ["openai-file-search: hosted collection-filtered path.", "local: build query vector and retrieve below.", "Local mode caches TF-IDF per collection."], "python")
    d.box(1050, 545, 330, 140, "Hosted path", ["Responses + file_search.", "Validate citations and answer.", "See question-answer diagram."], "shared")
    d.box(60, 785, 420, 210, "4A  Dense similarity", ["Cosine similarity over chunk vectors.", "Apply minimum dense score.", "Transformers / OpenAI: weight 1.0.", "Default hashing mode: weight 0.0.", "Hashing is not a semantic model."], "python")
    d.box(510, 785, 420, 210, "4B  Word TF-IDF", ["Fit vocabulary to collection chunks.", "Word unigrams + bigrams.", "Sublinear TF and cosine similarity.", "Keep positive lexical matches.", "Rank fusion weight: 1.0."], "python")
    d.box(960, 785, 420, 210, "4C  Character TF-IDF", ["Character n-grams of length 3-5.", "Query with near-miss terms only.", "Vocabulary similarity at least 0.84.", "Provides spelling tolerance.", "Rank fusion weight: 0.75."], "python")
    d.box(320, 1090, 800, 145, "5  Reciprocal rank fusion (RRF)", ["For each ranked source: fused score += weight / (10 + rank).", "Sort fused scores; return top-k chunks with document and source metadata.", "Reuse a cache only while its chunk IDs match; retain at most 32 collections."], "python")
    d.box(320, 1290, 800, 145, "6  Generate an answer or extract relevant passages", ["No matches: return processing/no-evidence message with grounded: false.", "Otherwise use OpenAI if available; extractive fallback on absence/failure.", "Overview fallback uses collection context, section paths, and section outlines."], "python")
    d.box(320, 1490, 800, 180, "7  Recover missed evidence and finalize", ["Unsupported focused answer: retry extraction over all READY collection chunks.", "Recover citations by matching extracted sentences back to their source chunks.", "Still unsupported: return grounded: false and an empty source list.", "Otherwise return answer + citations + grounded: true; record audit outcome."], "success")
    d.save()


def access(png):
    d = Diagram("05-authentication-and-access", "05  Authentication and access workflow", "The same user-facing session and role model applies to both primary backends.", 1480, png)
    d.arrow([(720, 290), (720, 345)])
    d.arrow([(720, 485), (720, 540)])
    d.arrow([(1000, 415), (1050, 415)])
    d.arrow([(720, 680), (720, 730), (270, 730), (270, 785)])
    d.arrow([(720, 730), (720, 785)])
    d.arrow([(720, 730), (1170, 730), (1170, 785)])
    for x in (270, 720, 1170):
        d.arrow([(x, 995), (x, 1040), (720, 1040), (720, 1095)])
    d.arrow([(720, 1240), (720, 1295)])
    d.box(320, 155, 800, 135, "1  Bootstrap CSRF protection", ["GET /api/csrf -> token + headerName + parameterName.", "Browser receives XSRF-TOKEN cookie; submits token with state-changing calls.", "Health, CSRF bootstrap, and API documentation are publicly accessible."], "shared")
    d.box(440, 345, 560, 140, "2  Sign in", ["POST /login with username, password, and CSRF.", "Validate environment-provisioned account.", "Java verifies BCrypt-backed account credentials."], "shared")
    d.box(1050, 345, 330, 140, "Access denied", ["Bad credentials -> HTTP 401.", "Bad CSRF -> HTTP 403.", "No authorized action runs."], "error")
    d.box(440, 540, 560, 140, "3  Establish and use a session", ["JSESSIONID identifies the authenticated session.", "Check authentication on protected API routes.", "Check CSRF for POST / DELETE / other mutations."], "shared")
    d.box(60, 785, 420, 210, "VIEWER", ["List collections and documents.", "Ask questions about collections.", "Inspect answers and source cards.", "Cannot upload or delete documents.", "Cannot access administrator audit."], "shared")
    d.box(510, 785, 420, 210, "EDITOR", ["All viewer operations.", "Create collections.", "Upload and retry documents.", "Delete individual documents.", "Cannot delete whole collections."], "shared")
    d.box(960, 785, 420, 210, "ADMIN", ["All editor operations.", "Delete entire collections.", "Read GET /api/admin/audit.", "Audit endpoint is cursor-paginated.", "Accounts configured via environment."], "shared")
    d.box(320, 1095, 800, 145, "4  Scope data operations and record outcomes", ["Collection IDs constrain retrieval, checksum lookups, and document operations.", "Audit records actor/action/outcome/resource; omit question and document text.", "Collection access is instance-wide across authenticated accounts."], "success")
    d.box(60, 1295, 1320, 115, "5  Sign out", ["POST /logout with CSRF -> invalidate session and remove the session cookie.", "Subsequent protected calls require authentication again."], "shared")
    d.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="Also render PNGs using Pillow and Windows Segoe UI fonts")
    args = parser.parse_args()
    for build in (architecture, ingestion, questions, python_retrieval, access):
        build(args.png)
