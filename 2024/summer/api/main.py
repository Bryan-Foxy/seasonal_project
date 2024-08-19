import os
import json
from ner import NER
from ocr import OCR
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

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
        #image = file.read()
        filename = secure_filename(file.filename)
        #filepath = os.path.join('uploads', filename)
        file.save(filename)
        # OCR
        ocr = OCR(filename, lang = ['fr'])
        result_ocr = ocr.detection()

        # NER
        tempory_path = "temporary.json"
        with open(tempory_path, "w") as f:
            json.dump(result_ocr, f, indent = 4)

        ner = NER(tempory_path)
        result_ner = ner.detection()
        os.remove(filename)
        os.remove(tempory_path)
        return jsonify({"result_ner": result_ner})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        

if __name__ == "__main__":
    app.run(debug = True)
    # Test with curl
    # curl -X POST -F file=@modele-facture-fr-classique-blanc-750px.png http://127.0.0.1:5000/ocr_and_ner