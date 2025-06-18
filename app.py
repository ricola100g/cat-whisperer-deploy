from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import base64
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-pro-vision")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    image_b64 = data.get("image_base64")
    prompt = data.get("prompt", "這隻貓現在的心情是什麼？請用可愛語氣描述")

    if not image_b64:
        return jsonify({"result": "❌ 沒有收到圖片"}), 400

    try:
        header, b64data = image_b64.split(",", 1)
        image_data = base64.b64decode(b64data)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"result": "❌ 圖片解析失敗"}), 400

    try:
        response = model.generate_content([prompt, image])
        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"result": "❌ Gemini 回應失敗"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)