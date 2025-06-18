from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import base64
import traceback
from io import BytesIO
import os
from rag_helper import KnowledgeRAG

app = Flask(__name__)
CORS(app)

# Load knowledge
rag = KnowledgeRAG("knowledge20250618.json")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/")
def home():
    return send_from_directory("dist", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        image_b64 = data.get("image_base64")
        prompt = data.get("prompt", "請描述這隻貓的情緒與意圖")

        _, b64data = image_b64.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(b64data))).convert("RGB")

        extracted = model.generate_content(["請描述這張貓咪圖片中可見的動作、肢體語言、表情與可能環境。請用繁體中文條列特徵：", image])
        features = extracted.text.replace("*", "-")

        matched = rag.query(features, k=4)
        reference = "\n".join([f"- {m}" for m in matched])

        final_prompt = f"""根據以下情境特徵，從貓行為學角度以擬人化方式描述這隻貓的情緒與想法：
可見特徵：
{features}

相關知識：
{reference}

請用繁體中文描述。若情緒偏負面，語氣應嚴肅而鼓勵；否則使用甜美俏皮風格，像獸醫師與主人解釋。
只需輸出一段自然、口語化的對話式文字即可。
"""
        reply = model.generate_content(final_prompt)
        return jsonify({"result": reply.text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"result": f"❌ 錯誤：{str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
