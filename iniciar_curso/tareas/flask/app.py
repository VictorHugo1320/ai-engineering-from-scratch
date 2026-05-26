from importlib.metadata import version
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)


@app.route("/")
def root():
    return jsonify({
        "service": "ai-dev flask demo",
        "versions": {
            "torch": version("torch"),
            "flask": version("flask"),
            "qdrant_client": version("qdrant-client"),
        },
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/sum")
def gpu_sum():
    a = float(request.args.get("a", 1.0))
    b = float(request.args.get("b", 2.0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = (torch.tensor([a], device=device) + torch.tensor([b], device=device)).item()
    return jsonify({"a": a, "b": b, "sum": result, "device": device})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)