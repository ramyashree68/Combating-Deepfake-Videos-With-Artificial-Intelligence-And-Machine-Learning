from flask import Flask, request, jsonify
from deepfake_detection import detect_deepfake  # Import your deepfake detection function

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Save the file and process it
        file_path = f"./uploads/{file.filename}"
        file.save(file_path)
        result = detect_deepfake(file_path)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
