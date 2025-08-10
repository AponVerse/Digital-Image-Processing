from flask import Flask, render_template, request, send_from_directory
import os
from enhancer import process_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['image']
    if file:
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, "enhanced_" + file.filename)
        file.save(input_path)

        _, enhanced_img = process_image(input_path)
        import cv2
        cv2.imwrite(output_path, enhanced_img)

        return render_template("index.html", result_url=output_path)
    return "No file uploaded!"

if __name__ == "__main__":
    app.run(debug=True)
