from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import base64, os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/")
def index():
    return send_from_directory("dist", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        _, b64data = data["image_base64"].split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(b64data)))
        vision_prompt = ["請分析貓的姿態、環境與情緒", image]
        res = model.generate_content(vision_prompt)
        return jsonify({"result": res.text})
    except Exception as e:
        return jsonify({"result": f"❌ 錯誤：{str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
