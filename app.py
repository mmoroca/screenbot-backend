from flask import Flask, request, jsonify
import requests
import base64
import os

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/")
def home():
    return "Servidor activo 🧠"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"error": "No image received"}), 400

    try:
        image_bytes = base64.b64decode(image_b64.split(",")[1])  # Elimina encabezado "data:image/png;base64,..."
    except Exception as e:
        return jsonify({"error": "Decoding failed", "details": str(e)}), 400

    response = requests.post(HF_API_URL, headers=headers, files={"file": image_bytes})

    if response.status_code == 200:
        result = response.json()
        caption = result[0].get("generated_text", "No se pudo generar una descripción.")
        return jsonify({"idea": f"¿Sabías? Parece que estás viendo: {caption} 🤔"})
    else:
        return jsonify({"error": "Error en Hugging Face", "details": response.text}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)