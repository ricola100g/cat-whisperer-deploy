from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
openai.api_key = "sk-..."  # Replace with your secure method

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_url = data.get("image_base64")
    prompt = data.get("prompt")

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        temperature=0.7
    )
    result = response.choices[0].message.content
    return jsonify({"result": result})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)