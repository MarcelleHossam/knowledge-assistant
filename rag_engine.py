"""
Core RAG engine: chunking, a small pure-python BM25 index, source
ingestion (Wikipedia + uploaded files), and LLM calls (streaming).

No heavyweight ML dependencies required -- everything here runs with
just `requests` (and optionally `pypdf` for PDF uploads).
"""
import os
import re
import math
import time
import uuid
import json
import requests
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

STOPWORDS = set("""
a an the of and or to in on for with is are was were be been being this that
these those it its as at by from into about over under again further than
then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t can will just
don should now i you he she we they them his her their our your what which
who whom
""".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def chunk_text(text, source_label, chunk_size=900, overlap=150):
    """Split text into overlapping character chunks on sentence-ish boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to end on a sentence boundary
        if end < n:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.4:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


class BM25Index:
    """Minimal BM25-Okapi implementation, no external deps."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []          # list of dicts: {id, text, source, tokens}
        self.doc_freq = Counter()
        self.avgdl = 0.0

    def add(self, doc_id, text, source):
        tokens = tokenize(text)
        self.docs.append({"id": doc_id, "text": text, "source": source, "tokens": tokens})
        for term in set(tokens):
            self.doc_freq[term] += 1
        self._recompute_avgdl()

    def remove_source(self, source):
        self.docs = [d for d in self.docs if d["source"] != source]
        self.doc_freq = Counter()
        for d in self.docs:
            for term in set(d["tokens"]):
                self.doc_freq[term] += 1
        self._recompute_avgdl()

    def _recompute_avgdl(self):
        if self.docs:
            self.avgdl = sum(len(d["tokens"]) for d in self.docs) / len(self.docs)
        else:
            self.avgdl = 0.0

    def _idf(self, term):
        n = len(self.docs)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def search(self, query, top_k=5):
        if not self.docs:
            return []
        q_tokens = tokenize(query)
        scores = []
        for d in self.docs:
            tf = Counter(d["tokens"])
            dl = len(d["tokens"]) or 1
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self._idf(term)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / (denom or 1)
            scores.append((score, d))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = [s for s in scores if s[0] > 0][:top_k]
        if not top:
            top = scores[:top_k]
        max_score = max((s for s, _ in top), default=1) or 1
        results = []
        for score, d in top:
            results.append({
                "chunk": d["text"],
                "source": d["source"],
                "score": min(score / (max_score * 1.15 + 1e-9), 1.0),
            })
        return results


class RAGEngine:
    def __init__(self):
        self.index = BM25Index()
        self.sources = {}  # id -> {id, label, type, chunk_count, added_at}

    # ---------------- status ----------------
    def status(self):
        return {
            "chunks": len(self.index.docs),
            "semantic_available": False,
            "sources": list(self.sources.values()),
            "llm_provider": llm_provider_name(),
            "cache_backend": "in-memory",
        }

    # ---------------- ingestion ----------------
    def add_source(self, label, source_type, full_text):
        src_id = uuid.uuid4().hex[:10]
        chunks = chunk_text(full_text, label)
        for c in chunks:
            self.index.add(src_id, c, src_id)
        self.sources[src_id] = {
            "id": src_id,
            "label": label,
            "type": source_type,
            "chunk_count": len(chunks),
            "added_at": time.time() * 1000,
        }
        return src_id, len(chunks)

    def remove_source(self, src_id):
        if src_id in self.sources:
            self.index.remove_source(src_id)
            del self.sources[src_id]
            return True
        return False

    def ingest_wikipedia(self, topic):
        topic = self._extract_title(topic)
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "format": "json", "prop": "extracts",
                    "explaintext": 1, "redirects": 1, "titles": topic,
                },
                headers={
                    "User-Agent": "UniversalKnowledgeAssistant/1.0 "
                                  "(https://github.com/example/knowledge-assistant; "
                                  "contact@example.com) python-requests"
                },
                timeout=(5, 15),  # (connect timeout, read timeout)
            )
        except requests.exceptions.ConnectionError:
            raise ValueError(
                "Could not connect to Wikipedia. This machine's internet "
                "access may be blocked by a firewall/antivirus for this "
                "app, or there's no internet connection right now."
            )
        except requests.exceptions.Timeout:
            raise ValueError("Wikipedia took too long to respond. Please try again.")
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)
        if not page or "extract" not in page or not page["extract"].strip():
            raise ValueError(f'No Wikipedia article found for "{topic}".')
        text = page["extract"]
        label = page.get("title", topic)
        return self.add_source(label, "wikipedia", text)

    @staticmethod
    def _extract_title(raw):
        """Accept either a plain topic ('Quantum mechanics') or a full
        Wikipedia URL (https://en.wikipedia.org/wiki/Quantum_mechanics)."""
        raw = raw.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            path = raw.split("/wiki/")[-1] if "/wiki/" in raw else raw.rsplit("/", 1)[-1]
            path = path.split("#")[0].split("?")[0]
            import urllib.parse
            path = urllib.parse.unquote(path)
            return path.replace("_", " ")
        return raw

    def ingest_file(self, filename, raw_bytes):
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "pdf":
            text = extract_pdf_text(raw_bytes)
        else:
            text = raw_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            raise ValueError("Could not extract any text from that file.")
        return self.add_source(filename, ext, text)

    # ---------------- retrieval ----------------
    def retrieve(self, question, top_k=5):
        return self.index.search(question, top_k=top_k)


def extract_pdf_text(raw_bytes):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed; run: pip install pypdf")
    import io
    reader = PdfReader(io.BytesIO(raw_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# =====================================================================
# LLM CALL (streaming). Supports Fireworks, OpenAI, or Anthropic via
# env vars. Falls back to a simple extractive "answer" (no API key
# needed) so the app still works end-to-end out of the box.
# =====================================================================

def llm_provider_name():
    if os.environ.get("FIREWORKS_API_KEY"):
        return "fireworks"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "extractive-fallback (no API key set)"


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{i+1}] (source: {c['source']})\n{c['chunk']}" for i, c in enumerate(chunks))
    return (
        "You are a careful research assistant. Answer the question using ONLY the "
        "numbered context snippets below. Cite claims inline like [1], [2]. If the "
        "context doesn't contain the answer, say so plainly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def stream_answer(question, chunks, temperature=0.0):
    """Yields text deltas. Tries real providers first, falls back to
    a simple extractive stitch of the top chunks."""
    prompt = build_prompt(question, chunks)

    if os.environ.get("FIREWORKS_API_KEY"):
        yield from _stream_fireworks(prompt, temperature)
        return
    if os.environ.get("OPENAI_API_KEY"):
        yield from _stream_openai(prompt, temperature)
        return
    if os.environ.get("ANTHROPIC_API_KEY"):
        yield from _stream_anthropic(prompt, temperature)
        return

    # Fallback: no key configured. Give a usable, clearly-labeled answer
    # built directly from the retrieved chunks so the demo still works.
    if not chunks:
        text = "I don't have any loaded sources to answer from yet."
    else:
        best = chunks[0]
        snippet = best['chunk'][:600].rstrip()
        text = (
            "⚠️ No LLM API key is configured, so I can't actually reason about "
            "this document — I can only show you the most relevant raw passage "
            "found by keyword search. Set FIREWORKS_API_KEY, OPENAI_API_KEY, or "
            f"ANTHROPIC_API_KEY in your .env file for real answers.\n\n"
            f"Most relevant passage [1]: {snippet}"
        )
    for i in range(0, len(text), 12):
        yield text[i:i + 12]
        time.sleep(0.01)


def _stream_fireworks(prompt, temperature):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}", "Content-Type": "application/json"}
    body = {
        "model": os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=body, stream=True, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            except Exception:
                continue
            if delta:
                yield delta


def _stream_openai(prompt, temperature):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    body = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=body, stream=True, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            except Exception:
                continue
            if delta:
                yield delta


def _stream_anthropic(prompt, temperature):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    with requests.post(url, headers=headers, json=body, stream=True, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            try:
                evt = json.loads(payload)
            except Exception:
                continue
            if evt.get("type") == "content_block_delta":
                delta = evt.get("delta", {}).get("text", "")
                if delta:
                    yield delta
