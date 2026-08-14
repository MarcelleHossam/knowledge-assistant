# Universal Knowledge Assistant

Your frontend (`templates/index.html`) alone couldn't work because it calls
a backend that didn't exist (`/api/status`, `/api/ingest/wikipedia`,
`/api/upload`, `/api/sources/:id`, `/api/chat_stream`). This project adds
that backend — a small Flask app + a pure-Python RAG engine (chunking,
BM25 search, Wikipedia/file ingestion, streaming LLM answers) — so the page
actually works.

## 1. Project structure

```
knowledge-assistant/
├── app.py              # Flask server, defines the /api/* routes
├── rag_engine.py        # chunking + BM25 index + ingestion + LLM streaming
├── templates/
│   └── index.html        # your frontend (Flask serves this at "/")
├── data/                 # empty at rest, kept out of git
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Flask's default behavior is to look for HTML in a folder called
`templates/`, which is why your file needs to live at
`templates/index.html` — that's the one change needed to your file itself
(everything else about it works as-is).

## 2. Run it locally

```bash
cd knowledge-assistant
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then edit .env, see step 3
python3 app.py
```

Open **http://127.0.0.1:5000** — that's it. The frontend and backend are
served from the same Flask app, so there's no separate "run the frontend"
step.

## 3. Getting "Failed to fetch" when loading a Wikipedia topic?

This means your browser sent the request fine, but the Python server
couldn't reach `en.wikipedia.org` — almost always Windows Firewall or
antivirus silently blocking outbound internet access for `python.exe`.

Fix:
1. Open **Windows Security** → **Firewall & network protection** → **Allow an app through firewall**
2. Find `python.exe` (or add it) and make sure both **Private** and **Public** are checked
3. Restart `python app.py` and try again

If you'd rather not touch firewall settings, file uploads (`.txt`/`.md`/`.pdf`) don't need internet access at all — try that instead to confirm the rest of the app works.

If the request now fails with a *clear error message* instead of "Failed to fetch" (e.g. "Could not connect to Wikipedia..."), that's actually working as designed — the backend now catches connection failures and reports them instead of hanging.

## 4. Add an LLM API key (optional but recommended)

Open `.env` and set **one** of:

```
FIREWORKS_API_KEY=your_key_here
```
or
```
OPENAI_API_KEY=your_key_here
```
or
```
ANTHROPIC_API_KEY=your_key_here
```

If you leave all of them blank, the app still runs end-to-end — it just
answers with the top retrieved passage instead of a generated answer, so
you can confirm everything is wired up before paying for an API key.

## 5. Try it

1. In the sidebar, type a topic (e.g. `Quantum mechanics`) and click **Load** — or drag a `.pdf` / `.txt` / `.md` file onto the dropzone.
2. Ask a question in the chat box at the bottom.
3. You'll see retrieval → generating stages, a streamed answer with `[1]` citations, and an expandable "Retrieved context" panel underneath.

## 6. What to upload to GitHub

Upload **everything except** what's in `.gitignore` (git will handle this
automatically once you push — you don't need to manually exclude files).
Concretely, this means you commit:

```
app.py
rag_engine.py
requirements.txt
.env.example
.gitignore
README.md
templates/index.html
data/.gitkeep
```

**Never commit your real `.env` file** — it holds your API key. Only
`.env.example` (with blank values) should go to GitHub.

### Pushing to GitHub

```bash
cd knowledge-assistant
git init
git add .
git commit -m "Universal Knowledge Assistant: Flask + RAG backend"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Anyone who clones the repo then does exactly step 2 + 3 above to run it.

## Notes on the implementation

- **Retrieval**: BM25 (a proven, dependency-free keyword-ranking algorithm) over overlapping ~900-character chunks. No embeddings/vector DB needed, so there's nothing extra to install or pay for. `semantic_available` in the status panel is `false` for this reason — you can swap in an embeddings-based retriever later without changing the frontend contract.
- **Wikipedia ingestion**: uses Wikipedia's public `action=query&prop=extracts` API — no key required.
- **File ingestion**: `.txt` / `.md` read directly; `.pdf` parsed with `pypdf`.
- **Chat endpoint**: `/api/chat_stream` streams Server-Sent Events (`stage`, `retrieval`, `answer`, `done`, `error`) exactly matching what your frontend's `sendQuestion()` already expects.
- **State**: in-memory only (matches the original frontend's comment about this) — restarting `app.py` clears loaded sources. Good enough for local/demo use; swap in a real DB if you need persistence across restarts.
