from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import os
import requests

HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "image" not in data:
        print("❌ No image received in request")
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = data["image"].split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        print("✅ Image received and processed")

        response = requests.post(
            "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
            headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
            files={"file": ("image.png", buffered.getvalue(), "image/png")}
        )

        result = response.json()
        print("🧠 HuggingFace response:", result)
        caption = result[0]["generated_text"] if isinstance(result, list) else "No se pudo analizar la imagen."
        return jsonify({"idea": caption})
    except Exception as e:
        print("🔥 Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
