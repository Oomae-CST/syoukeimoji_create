#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# ── API キー ───────────────────────────────────────────
api_key = "APIキーを入れてください"
client = OpenAI(api_key=api_key)
app = Flask(__name__)

# ── 画面 ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── /convert ─────────────────────────────────────────
@app.route("/convert", methods=["POST"])
def convert():
    if "image" not in request.files:
        return jsonify(error="no file"), 400
    img_bytes = request.files["image"].read()

    # ① GPT-4o-mini に「画像生成用プロンプト + ラベル」を作らせる
    data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()

    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {"role": "system",
                 "content": (
                     "You are an expert prompt engineer for image generation. "
                     "Return ONLY valid JSON like "
                     "{\"prompt\":\"...\",\"label\":\"...\"}. "
                     "The prompt must include the Japanese word '象形文字' and instruct the generator to draw the object "
                     "as a single 象形文字 (black strokes, no fill, "
                     "transparent background, <=20 English words). "
                     "label = short English noun of the object."
                 )},
                {"role": "user",
                 "content": [
                     {"type": "text",
                      "text": "Convert this photo into a single pictogram. "
                              "Analyse the object and craft the prompt."},
                     {"type": "image_url", "image_url": {"url": data_uri}}
                 ]
                }
            ],
            response_format={"type": "json_object"}
        )
        obj = json.loads(chat.choices[0].message.content)
        prompt = obj.get("prompt") or "A simple black-stroke pictogram."
        label  = obj.get("label") or "object"
    except Exception as e:
        print("[prompt-gen error]", e)
        prompt = ("A simple black-stroke pictogram of an unknown object, "
                  "transparent background, no shading.")
        label = "unknown"

    # ② 画像生成 ─ gpt-image-1 → dall-e-3 → dall-e-2
    for model in ("gpt-image-1", "dall-e-3", "dall-e-2"):
        try:
            gen = client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size="512x512",
                response_format="b64_json"
            )
            b64_png = gen.data[0].b64_json
            break
        except Exception as e:
            print(f"[{model} failed] {e}")
    else:
        return jsonify(error="all models failed"), 500

    return jsonify(
        image="data:image/png;base64," + b64_png,
        label=label,
        prompt=prompt,
        model_used=model
    )

if __name__ == "__main__":
    app.run(debug=True)