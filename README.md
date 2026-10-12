# Universal Knowledge Assistant


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



