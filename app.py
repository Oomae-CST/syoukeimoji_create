#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import base64
import imghdr
from pathlib import Path
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from openai import OpenAI
from werkzeug.utils import secure_filename

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OpenAI APIキーが設定されていません")

client = OpenAI(api_key=api_key)
app = Flask(__name__)

def build_oracle_bone_prompt(features: list[str]) -> str:
    if not features:
        core = "A single simple oracle-bone-style glyph"
    elif len(features) == 1:
        core = f"A single oracle-bone-style glyph representing {features[0]}"
    else:
        *rest, last = features
        core = "An oracle-bone-style glyph featuring " + ", ".join(rest) + f", and {last}"

    positive_style = (
        " in authentic Shang/Zhou dynasty oracle bone script aesthetics: "
        "carved incision-like monoline strokes, slightly irregular edges as if engraved, "
        "square-ish composition, clear silhouette, highly simplified pictography, "
        "no perspective, no shading, no texture fill, no background. "
        "Use only flat black strokes on a plain, empty canvas. "
        "Design must be drawable fast with few strokes and readable at small sizes."
    )

    negative_constraints = (
        " Avoid modern Chinese font styles (song/hei/kai), avoid calligraphy brush effects, "
        "no gradients, no colors, no photos, no shadows, no decorations, no border, "
        "no Latin/Arabic letters, no Arabic numerals, no text, no English words."
    )

    return core + positive_style + negative_constraints

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/features", methods=["POST"])
def features():
    img_file = request.files.get("image")
    if not img_file or img_file.filename == "":
        return jsonify(error="ファイルがアップロードされていません"), 400

    img_bytes = img_file.read()
    ext = imghdr.what(None, h=img_bytes) or "png"
    data_uri = f"data:image/{ext};base64," + base64.b64encode(img_bytes).decode()

    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert vision analyst. "
                        "Analyze the provided image carefully and output ONLY valid JSON in the form "
                        "{\"features\":[\"...\", \"...\", ...]}. "
                        "The list must capture the subject (e.g. cat, tree, person), "
                        "its main parts (e.g. ears, branches, limbs), "
                        "salient shapes (e.g. round eyes, triangular ears, tall trunk), "
                        "textures (e.g. smooth, rough, furry, leafy), "
                        "and visible colors if distinctive (e.g. green leaves, brown bark). "
                        "Focus on the core recognizable traits of the object(s) in the photo. "
                        "Do not include extra explanations, only JSON."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            response_format={ "type": "json_object" }
        )
        raw = chat.choices[0].message.content
        if isinstance(raw, str):
            obj = json.loads(raw)
        else:
            obj = raw
        features_list = obj.get("features", [])
        if not isinstance(features_list, list):
            features_list = []
    except Exception as e:
        print("[feature-extract error]", e)
        features_list = []

    return jsonify(features=features_list)

@app.route("/convert", methods=["POST"])
def convert():
    img_file = request.files.get("image")
    if not img_file or img_file.filename == "":
        return jsonify(error="ファイルがアップロードされていません"), 400

    features = request.form.getlist("features")
    if not features:
        return jsonify(error="特徴が選択されていません"), 400

    prompt_for_image = build_oracle_bone_prompt(features)
    print("Prompt for Image:", prompt_for_image)

    model_sizes = {
        "gpt-image-1": "1024x1024",
        "dall-e-3": "1024x1024",
        "dall-e-2": "512x512"
    }
    b64_png = None
    model_used = None

    for model, size in model_sizes.items():
        try:
            if model == "gpt-image-1":
                gen = client.images.generate(
                    model=model,
                    prompt=prompt_for_image,
                    n=1,
                    size=size
                )
                candidate = getattr(gen.data[0], "b64_json", None)
                if not candidate:
                    url = getattr(gen.data[0], "url", None)
                    if url:
                        resp = requests.get(url, timeout=30)
                        resp.raise_for_status()
                        candidate = base64.b64encode(resp.content).decode()
            else:
                gen = client.images.generate(
                    model=model,
                    prompt=prompt_for_image,
                    n=1,
                    size=size,
                    response_format="b64_json"
                )
                candidate = gen.data[0].b64_json

            if candidate:
                b64_png = candidate
                model_used = model
                break
        except Exception as e:
            print(f"[{model} failed]", e)
            continue

    if not b64_png:
        return jsonify(error="全てのモデルで生成失敗"), 500

    out_dir = Path("static/downloads")
    out_dir.mkdir(exist_ok=True, parents=True)
    filename = secure_filename(f"{model_used}_oraclebone_{Path(img_file.filename).stem}.png")
    filepath = out_dir / filename
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_png))

    return jsonify(
        image="data:image/png;base64," + b64_png,
        features=features,
        prompt=prompt_for_image,
        model_used=model_used,
        download_url=f"/download/{filename}"
    )

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory("static/downloads", filename, as_attachment=True)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
