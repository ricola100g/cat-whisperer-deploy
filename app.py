from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import base64
from PIL import Image
from io import BytesIO
import os
import traceback

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/")
def index():
    return send_from_directory("dist", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    image_b64 = data.get("image_base64")
    prompt = data.get("prompt", "請描述這隻貓的情緒")

    if not image_b64:
        return jsonify({"result": "❌ 沒有收到圖片"}), 400

    try:
        header, b64data = image_b64.split(",", 1)
        image_data = base64.b64decode(b64data)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"result": f"❌ 圖片處理錯誤：{str(e)}"}), 400

    try:
        response = model.generate_content([prompt, image])
        return jsonify({"result": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"result": f"❌ Gemini 回應失敗：{str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
