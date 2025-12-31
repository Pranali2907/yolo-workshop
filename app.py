# ============================================================
# © 2025 AreneSha AI Labs
#
# Proprietary and Confidential
# ============================================================

from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import uuid
from ultralytics import YOLO

app = Flask(__name__)

OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global webcam state
model = None
camera = None
streaming = False


@app.route("/")
def index():
    return render_template("index.html")


# ---------------- IMAGE INFERENCE ----------------
@app.route("/run_image_inference", methods=["POST"])
def run_image_inference():
    model_file = request.files.get("model")
    image_file = request.files.get("image")

    if not model_file or not image_file:
        return jsonify({"error": "Model or image missing"}), 400

    model_path = os.path.join(OUTPUT_DIR, model_file.filename)
    image_path = os.path.join(OUTPUT_DIR, image_file.filename)

    model_file.save(model_path)
    image_file.save(image_path)

    yolo = YOLO(model_path)
    results = yolo(image_path)

    output_name = f"result_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    annotated = results[0].plot(labels=True, conf=True)
    cv2.imwrite(output_path, annotated)

    return jsonify({
        "result_image": f"/static/outputs/{output_name}"
    })


# ---------------- LIVE WEBCAM STREAM ----------------
def generate_frames():
    global camera, model, streaming

    while streaming:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        annotated = results[0].plot(labels=True, conf=True)

        _, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/start_webcam", methods=["POST"])
def start_webcam():
    global model, camera, streaming

    model_file = request.files.get("model")
    if not model_file:
        return jsonify({"error": "Model missing"}), 400

    model_path = os.path.join(OUTPUT_DIR, model_file.filename)
    model_file.save(model_path)

    model = YOLO(model_path)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return jsonify({"error": "Webcam not accessible"}), 500

    streaming = True
    return jsonify({"status": "Webcam started"})


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop_webcam", methods=["POST"])
def stop_webcam():
    global streaming, camera

    streaming = False
    if camera:
        camera.release()
        camera = None

    return jsonify({"status": "Webcam stopped"})


if __name__ == "__main__":
    app.run(debug=True)

