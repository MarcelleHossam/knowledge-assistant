import json
import time
from flask import Flask, request, jsonify, Response, render_template
from rag_engine import RAGEngine, stream_answer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; .env vars just won't be auto-loaded

app = Flask(__name__)
engine = RAGEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify(engine.status())


@app.route("/api/ingest/wikipedia", methods=["POST"])
def ingest_wikipedia():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Please provide a topic."}), 400
    try:
        src_id, n_chunks = engine.ingest_wikipedia(topic)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"id": src_id, "chunks": n_chunks})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400
    allowed = (".pdf", ".txt", ".md", ".markdown")
    if not f.filename.lower().endswith(allowed):
        return jsonify({"error": "Only PDF, TXT, or MD files are supported."}), 400
    try:
        src_id, n_chunks = engine.ingest_file(f.filename, f.read())
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"id": src_id, "chunks": n_chunks})


@app.route("/api/sources/<src_id>", methods=["DELETE"])
def delete_source(src_id):
    ok = engine.remove_source(src_id)
    return jsonify({"deleted": ok})


@app.route("/api/chat_stream", methods=["POST"])
def chat_stream():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    advanced = bool(data.get("advanced", True))
    temperature = float(data.get("temperature", 0.0))
    threshold = float(data.get("threshold", 0.2))

    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def generate():
        if not question:
            yield sse({"type": "error", "message": "Empty question."})
            return

        yield sse({"type": "stage", "stage": "retrieving"})
        chunks = engine.retrieve(question, top_k=5)

        if advanced and chunks:
            yield sse({"type": "stage", "stage": "reranking"})
            chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)

        top_score = chunks[0]["score"] if chunks else 0.0
        yield sse({"type": "retrieval", "top_score": top_score})

        if not chunks:
            yield sse({"type": "stage", "stage": "generating"})
            msg = ("There's nothing loaded yet — load a Wikipedia topic or "
                   "upload a document on the left first.")
            for i in range(0, len(msg), 12):
                yield sse({"type": "answer", "delta": msg[i:i + 12]})
                time.sleep(0.01)
            yield sse({"type": "done", "chunks": []})
            return

        yield sse({"type": "stage", "stage": "generating"})
        try:
            for delta in stream_answer(question, chunks, temperature=temperature):
                yield sse({"type": "answer", "delta": delta})
        except Exception as e:
            yield sse({"type": "error", "message": f"LLM call failed: {e}"})
            return

        yield sse({"type": "done", "chunks": chunks})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


if __name__ == "__main__":
    # debug=False avoids the auto-reloader killing in-flight requests
    # (e.g. a slow Wikipedia fetch) with a generic "Failed to fetch" in
    # the browser. Set FLASK_DEBUG=1 in your environment if you want
    # debug mode back while developing.
    import os
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
