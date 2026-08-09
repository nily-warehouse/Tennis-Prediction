from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import estimator


BASE_DIR = Path(__file__).resolve().parent
STATIC_FILES = {"app.js", "model-adapter.js", "players.js", "styles.css"}

app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:filename>")
def static_file(filename):
    if filename not in STATIC_FILES:
        return jsonify(error="Not found"), 404
    return send_from_directory(BASE_DIR, filename)


@app.post("/api/estimate")
def estimate():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Request body must be a JSON object"), 400
        return jsonify(estimator.estimate(payload))
    except (KeyError, TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        app.logger.exception("Estimation failed")
        return jsonify(error=str(error)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
