from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/highlight": {"origins": "*"}}, methods=["POST", "OPTIONS"])

NOTES_FILE = "notes.txt"

@app.route("/highlight", methods=["POST", "OPTIONS"])
def save_highlight():
    if request.method == "OPTIONS":
        return '', 200  # Handle CORS preflight

    data = request.json
    note = data['text'].strip()

    if note:
        with open(NOTES_FILE, "a") as f:
            wrapped = f"~\n{note}\n~"
            f.write("\n\n" + wrapped)

    return {"status": "saved"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
