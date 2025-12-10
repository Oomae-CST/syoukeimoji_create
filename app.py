#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import base64
import imghdr
from pathlib import Path
import io
import random
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from openai import OpenAI
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageFilter, ImageChops

# --- 実行前に必ず環境変数を設定してください ---
# 例:  export OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXXXXXX"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY が設定されていません")

client = OpenAI(api_key=api_key)
app = Flask(__name__)

DOWNLOAD_DIR = Path("static/downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ===== 甲骨文字（象形）プロンプト =====
def build_oracle_bone_prompt(features):
    """
    画像から抽出された英語の一般名詞（hypernyms）features をもとに、
    甲骨文（象形）スタイルのグリフを生成させるプロンプト。
    """
    if not features:
        core = "A single simple oracle-bone-style glyph"
    elif len(features) == 1:
        core = f"A single oracle-bone-style glyph representing {features[0]}"
    else:
        *rest, last = features
        core = "An oracle-bone-style glyph featuring " + ", ".join(rest) + f", and {last}"

    positive_style = (
        " in authentic Shang/Zhou dynasty oracle bone script (Jiaguwen) aesthetics: "
        "single-color carved incision strokes, slight roughness as if engraved into bone, "
        "rectilinear / square-ish composition, highly simplified pictography, "
        "stroke economy (very few strokes), clear silhouette, no perspective."
    )
    negative_constraints = (
        " Absolutely avoid illustration-like rendering: no shading, no gradients, "
        "no lighting, no perspective depth, no 3D, no textures or hatching fill, "
        "no decorative elements, no background, no color (black strokes only), "
        "no modern Chinese fonts (song/hei/kai), no calligraphy brush effects, "
        "no Latin/Arabic letters, no numerals, no English words, no surrounding frame."
    )
    rendering_rules = (
        " Use only flat black monoline strokes on a plain white canvas; "
        "center a single glyph; leave generous margins around it."
    )
    jp_hint = " 象形文字を生成してください。"

    return core + positive_style + negative_constraints + " " + rendering_rules + jp_hint

# ===== 象形ポストプロセス =====
def oracleize_png_bytes(png_bytes, size_hint=1024):
    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    im = ImageOps.autocontrast(im)
    th = 120 + random.randint(-10, 10)
    im_bin = im.point(lambda x: 0 if x < th else 255, mode="1").convert("L")
    im_eroded = im_bin.filter(ImageFilter.MinFilter(3))
    im_refined = im_eroded.filter(ImageFilter.MaxFilter(3))
    im_blur = im_refined.filter(ImageFilter.GaussianBlur(radius=0.6))
    th2 = 140 + random.randint(-10, 10)
    im_final = im_blur.point(lambda x: 0 if x < th2 else 255, mode="1").convert("L")
    out = io.BytesIO()
    im_final.save(out, format="PNG")
    return out.getvalue()

# ===== モーションブラー =====
def _motion_blur(im, size=5, angle="horizontal"):
    if size < 3:
        return im
    if angle == "horizontal":
        mat = [0.0] * (size * size)
        mid = size // 2
        for i in range(size):
            mat[mid * size + i] = 1.0 / size
    elif angle == "diag_pos":  # ↗
        mat = [0.0] * (size * size)
        for i in range(size):
            mat[(size - 1 - i) * size + i] = 1.0 / size
    else:  # "diag_neg" ↘
        mat = [0.0] * (size * size)
        for i in range(size):
            mat[i * size + i] = 1.0 / size
    return im.filter(ImageFilter.Kernel((size, size), mat, scale=sum(mat), offset=0))

# ===== 漢字用ポストプロセス =====
def kanjiify_png_bytes(png_bytes):
    """
    Bスタイル（現代楷書）向けの後処理
    """
    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    im = ImageOps.autocontrast(im)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.1, percent=140, threshold=3))
    mb_h = _motion_blur(im, size=5, angle="horizontal")
    mb_p = _motion_blur(im, size=5, angle="diag_pos")
    mb_n = _motion_blur(im, size=5, angle="diag_neg")
    im_mb = ImageChops.lighter(ImageChops.lighter(mb_h, mb_p), mb_n)
    th = 155
    im_bin = im_mb.point(lambda x: 0 if x < th else 255, mode="1").convert("L")
    im_bin = im_bin.filter(ImageFilter.MaxFilter(3))
    im_soft = im_bin.filter(ImageFilter.GaussianBlur(radius=0.3))
    th2 = 160
    im_final = im_soft.point(lambda x: 0 if x < th2 else 255, mode="1").convert("L")
    out = io.BytesIO()
    im_final.save(out, format="PNG")
    return out.getvalue()

def to_data_uri(image_bytes, ext_hint="png"):
    return f"data:image/{ext_hint};base64," + base64.b64encode(image_bytes).decode()

# ===== Vision で特徴抽出（共通ロジック） =====
def extract_features_core(image_bytes, ext="png"):
    """
    画像バイト列から features と、使用した System プロンプトを返す。
    features が取れなかった場合は ([], system_prompt) を返す。
    """
    data_uri = f"data:image/{ext};base64," + base64.b64encode(image_bytes).decode()

    features_system_prompt = (
        "You are an image tagging expert. "
        "List only the OBJECTS present in the image as English common nouns (hypernyms), in singular form. "
        'Return ONLY valid JSON of the form {"features":["...", "..."]}; no extra text. '
        "Strict rules: "
        "- Do NOT include brands, models, proper nouns, text content, colors, materials, states, actions, or abstract terms. "
        "- Use hypernyms: e.g., 'car' (not 'Toyota sedan'), 'dog' (not 'Shiba Inu'), 'cup' (not 'brand mug'). "
        "- People should be 'person'; animals as common nouns (e.g., 'dog', 'cat'). "
        "- Deduplicate; provide about 5–12 words. "
        "- Output in lowercase English; ASCII only."
    )

    cleaned = []

    try:
        chat = client.chat.completions.create(
            model="gpt-4o",  # 必要に応じて利用可能な Vision 対応モデルに変更
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": features_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and output JSON as instructed."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        }
                    ]
                },
            ],
        )

        raw = chat.choices[0].message.content
        print("[features raw content]", repr(raw))

        if isinstance(raw, str):
            obj = json.loads(raw)
        else:
            obj = {}

        features_list = obj.get("features", []) if isinstance(obj, dict) else []
        if not isinstance(features_list, list):
            features_list = []

        # 正規化処理
        seen = set()
        for f in features_list:
            if not isinstance(f, str):
                continue
            t = f.strip().lower()
            if not t:
                continue
            if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
                t = t[:-1]
            try:
                t.encode("ascii")
            except UnicodeEncodeError:
                continue
            if t not in seen:
                seen.add(t)
                cleaned.append(t)

        if len(cleaned) > 12:
            cleaned = cleaned[:12]

    except Exception as e:
        print("[feature-extract error]", repr(e))
        cleaned = []

    print("[extract_features_core] cleaned =", cleaned)
    return cleaned, features_system_prompt

# ===== 象形画像 → 構造抽出 =====
def extract_structure_from_image(image_bytes):
    data_uri = to_data_uri(image_bytes, "png")
    system_prompt = (
        "You are an expert glyph structure analyzer. "
        "Given an oracle-bone-style black monoline glyph image, "
        "output ONLY valid JSON describing its abstract structure for conversion into a kanji-like character. "
        "Use this JSON schema: {"
        '"strokes": ["vertical|horizontal|diagonal_up_right|diagonal_down_right|curve"...], '
        '"intersections": <integer>, '
        '"layout": "left-right|top-bottom|enclosure|single|other", '
        '"parts": ["short text fragments of salient parts"...], '
        '"stroke_count_hint": <integer between 4 and 15>'
        "}. "
        "No extra text; JSON only."
    )

    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this glyph image and output JSON as instructed."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        }
                    ]
                },
            ],
        )

        raw = chat.choices[0].message.content
        print("[structure raw content]", repr(raw))

        if isinstance(raw, str):
            obj = json.loads(raw)
        else:
            obj = {}

        if not isinstance(obj, dict):
            obj = {}

    except Exception as e:
        print("[structure-extract error]", repr(e))
        obj = {}

    return {"structure": (obj or {}), "structure_system_prompt": system_prompt}

# ===== ★象形の特徴も踏まえた 現代楷書スタイル用プロンプト =====
def build_kanji_prompt_from_structure_and_features(structure, semantic_features=None):
    """
    - structure: 象形画像から解析した構造情報（strokes/layout/parts など）
    - semantic_features: 元画像 → 象形で使った意味的な特徴（英語名詞のリスト）
      これを「意味上のヒント」として漢字設計に反映させる。
    """
    if semantic_features is None:
        semantic_features = []

    strokes = structure.get("strokes", [])
    layout = structure.get("layout", "single")
    parts = structure.get("parts", [])
    intersections = structure.get("intersections")
    stroke_count_hint = structure.get("stroke_count_hint")

    prompt = (
        "You are a specialist in modern Kaisho (regular script) kanji design. "
        "Redesign the glyph into a single kanji-like character based on BOTH:\n"
        "  (1) the abstract structural analysis of the oracle-bone pictograph, and\n"
        "  (2) the semantic features that the original pictograph was meant to represent.\n\n"
        "Your goal: create one plausible Kaisho-style character that visually echoes the oracle-bone structure\n"
        "while encoding the semantic intent suggested by the features. However, do NOT draw a pictorial logo:\n"
        "the result must look like a normal, legible kanji-form (no icons, no real-world illustrations).\n\n"
        "Kaisho style rules:\n"
        "- Black strokes only on plain white background.\n"
        "- Clear stroke categories: vertical, horizontal, diagonal, turning.\n"
        "- Terminals:\n"
        "  * Tome: firm blunt stop (slight corner softening).\n"
        "  * Hane: upward flick ~20–30 degrees near the end.\n"
        "  * Harai: sweeping taper over the last ~10–25% of the stroke length.\n"
        "- Subtle thickness variation is allowed only near terminals; main stroke bodies should appear nearly monoline.\n"
        "- Balanced spacing and square-ish overall proportion (like Kaisho), avoid decorative effects, texture, shading, or gradients.\n"
        "- Output one centered character.\n\n"
        "Structural hints from the analyzed oracle-bone glyph:\n"
    )
    if strokes:
        prompt += f"- stroke types present: {', '.join(strokes)}\n"
    if layout:
        prompt += f"- overall layout: {layout}\n"
    if parts:
        prompt += f"- salient subcomponents: {', '.join(parts[:6])}\n"
    if intersections is not None:
        prompt += f"- stroke intersections: {intersections}\n"
    if stroke_count_hint:
        prompt += f"- target total stroke count: {stroke_count_hint}\n"

    # ★ ここで、元々の features（意味的な特徴）をヒントとして追加
    if semantic_features:
        prompt += "\nSemantic intent hints (from the original object-image → oracle-bone pictograph):\n"
        prompt += (
            "- key nouns / concepts to encode (do NOT draw them literally as icons, only encode them via kanji-like components): "
            + ", ".join(semantic_features[:8])
            + "\n"
        )
        prompt += (
            "Try to choose radical-like components and overall balance so that the character could plausibly be associated\n"
            "with these concepts in a dictionary-like sense, while still feeling structurally related to the analyzed glyph.\n"
        )

    return prompt

# ===== 特徴抽出 API =====
@app.route("/features", methods=["POST"])
def features():
    img_file = request.files.get("image")
    if not img_file or img_file.filename == "":
        return jsonify(error="ファイルがアップロードされていません"), 400

    img_bytes = img_file.read()
    ext = imghdr.what(None, h=img_bytes) or "png"

    cleaned, features_system_prompt = extract_features_core(img_bytes, ext)

    print("[/features] cleaned features:", cleaned)
    print("[/features] prompt head:", features_system_prompt[:80], "...")

    return jsonify(
        features=cleaned,
        features_prompt=features_system_prompt,
        features_system_prompt=features_system_prompt,
    )

# ===== 画像生成（モデル分岐） =====
def generate_image_with_models(prompt_for_image, src_filename, postprocess="oracle"):
    model_sizes = {
        "gpt-image-1": "1024x1024",
        "dall-e-3": "1024x1024",
        "dall-e-2": "512x512",
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
                    size=size,
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
                    response_format="b64_json",
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
        raise RuntimeError("全てのモデルで生成失敗")

    raw_png_bytes = base64.b64decode(b64_png)
    if postprocess == "kanji":
        refined_png_bytes = kanjiify_png_bytes(raw_png_bytes)
    else:
        refined_png_bytes = oracleize_png_bytes(raw_png_bytes, size_hint=1024)

    filename = secure_filename(f"{model_used}_oraclebone_{Path(src_filename).stem}.png")
    filepath = DOWNLOAD_DIR / filename
    with open(filepath, "wb") as f:
        f.write(refined_png_bytes)

    return model_used, filename, refined_png_bytes

@app.route("/")
def index():
    return render_template("index.html")

# ===== 象形生成 =====
@app.route("/convert", methods=["POST"])
def convert():
    img_file = request.files.get("image")
    if not img_file or img_file.filename == "":
        return jsonify(error="ファイルがアップロードされていません"), 400

    img_bytes = img_file.read()
    ext = imghdr.what(None, h=img_bytes) or "png"
    src_filename = img_file.filename or "source.png"

    # フロントから来た features
    features_from_form = request.form.getlist("features")
    print("[convert] features from form:", features_from_form)

    features = features_from_form[:]

    # 何も来ていなければサーバー側で自動抽出
    if not features:
        auto_features, _ = extract_features_core(img_bytes, ext)
        print("[convert] auto extracted features:", auto_features)
        features = auto_features

    if not features:
        print("[convert] ERROR: features is still empty after auto extraction")
        return jsonify(error="特徴抽出に失敗しました（features が空です）"), 500

    oracle_prompt = build_oracle_bone_prompt(features)
    print("[convert] final features used:", features)
    print("[convert] Prompt for Image (oracle):", oracle_prompt)

    try:
        model_used, filename, refined_png_bytes = generate_image_with_models(
            oracle_prompt, src_filename, postprocess="oracle"
        )
    except Exception as e:
        print("[convert] image generation error:", e)
        return jsonify(error=str(e)), 500

    return jsonify(
        image="data:image/png;base64," + base64.b64encode(refined_png_bytes).decode(),
        features=features,
        oracle_prompt=oracle_prompt,
        model_used=model_used,
        oracle_image_id=filename,
        download_url=f"/download/{filename}",
    )

# ===== 象形 → 漢字生成 =====
@app.route("/kanji_from_oracle", methods=["POST"])
def kanji_from_oracle():
    oracle_image_id = request.form.get("oracle_image_id", "").strip()
    if not oracle_image_id:
        return jsonify(error="oracle_image_id がありません"), 400

    # ★ ここで features（意味的特徴）を受け取る
    semantic_features = request.form.getlist("features")
    print("[kanji_from_oracle] semantic features from form:", semantic_features)

    oracle_path = DOWNLOAD_DIR / secure_filename(oracle_image_id)
    if not oracle_path.exists():
        return jsonify(error="指定の象形画像が見つかりません"), 404

    oracle_bytes = oracle_path.read_bytes()
    struct_pack = extract_structure_from_image(oracle_bytes)
    structure = struct_pack.get("structure", {})
    structure_system_prompt = struct_pack.get("structure_system_prompt", "")

    # ★ 構造 + semantic_features を元に漢字プロンプトを生成
    kanji_prompt = build_kanji_prompt_from_structure_and_features(structure, semantic_features)
    print("[kanji_from_oracle] structure:", structure)
    print("[kanji_from_oracle] kanji_prompt:", kanji_prompt)

    try:
        model_used, filename, refined_png_bytes = generate_image_with_models(
            kanji_prompt, oracle_image_id, postprocess="kanji"
        )
    except Exception as e:
        print("[kanji_from_oracle] image generation error:", e)
        return jsonify(error=str(e)), 500

    return jsonify(
        image="data:image/png;base64," + base64.b64encode(refined_png_bytes).decode(),
        structure=structure,
        structure_system_prompt=structure_system_prompt,
        kanji_prompt=kanji_prompt,
        model_used=model_used,
        download_url=f"/download/{filename}",
    )

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(str(DOWNLOAD_DIR), filename, as_attachment=True)

if __name__ == "__main__":
    # 本番では debug=False 推奨
    app.run(debug=True, host="0.0.0.0", port=5000)
