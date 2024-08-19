from ner import NER
from ocr import OCR
from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route("/")
def home():
    return "Test API"

@app.route("/ocr_and_ner", methods = ['POST'])
def ocr_and_ner():
    if 'file' not in request.files:
        return jsonify({"error": "File not found"}), 400
    
    file = request.files['file']
    if file == "":
        return jsonify({"error": "File not found"}), 400
    
    try:
        image = file.read()
        # OCR
        ocr = OCR(image, lang = ['fr'])
        result_ocr = ocr.detection()

        # NER
        ner = NER(result_ocr)
        result_ner = ner.detection()
        return jsonify({"result_ocr": result_ocr, "result_ner": result_ner})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        

if __name__ == "__main__":
    app.run(debug = True)