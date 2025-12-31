from flask import Flask, render_template, request, jsonify
import os
import cv2
from ultralytics import YOLO
import uuid

app = Flask(__name__)

OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    model = YOLO(model_path)
    results = model(image_path)

    output_name = f"result_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    annotated = results[0].plot()
    cv2.imwrite(output_path, annotated)

    return jsonify({
        "result_image": f"/static/outputs/{output_name}"
    })

# ---------------- WEBCAM ----------------
@app.route("/run_webcam", methods=["POST"])
def run_webcam():
    model_file = request.files.get("model")

    if not model_file:
        return jsonify({"error": "Model missing"}), 400

    model_path = os.path.join(OUTPUT_DIR, model_file.filename)
    model_file.save(model_path)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return jsonify({"error": "Webcam not accessible"}), 500

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        for r in results:
            frame = r.plot()

        cv2.imshow("YOLOv8 Live Detection (Press Q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return jsonify({"status": "Webcam closed"})

if __name__ == "__main__":
    app.run(debug=True)
