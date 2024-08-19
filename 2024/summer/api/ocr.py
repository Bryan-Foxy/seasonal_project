import cv2
import json
import easyocr
import matplotlib.pyplot as plt

class OCR:
    def __init__(self, path, lang = 'en', gpu = False):
        super(OCR, self).__init__()
        self.path = path
        self.lang = lang
        self.gpu = gpu
    
    def _load_images(self):
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _plot(self, todetect, save = False):
        plt.title("Detection")
        plt.imshow(todetect)
        if save:
            plt.savefig("detection.png")
        plt.show()
    
    def detection(self, plot = False):
        print('Initialize the model')
        reader = easyocr.Reader(self.lang, gpu = self.gpu)
        todetect = self._load_images()
        result = reader.readtext(todetect, detail = 1, paragraph = False)
        detections = []

        with open("api/text_detect.txt", "w") as f:
            for (bbox, text, prob) in result:
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[0]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                print(f"Probability: {prob:.3f}")
                text =  "".join([c if ord(c) < 128 else "" for c in text]).strip()
                detections.append({
                    "text": text,
                    "prob": prob,
                    "bbox": {"tl": tl, "tr": tr, "br": br, "bl": bl}
                })
                cv2.rectangle(todetect, tl, br, (0, 255, 0), 2)
                cv2.putText(todetect, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print(text, file = f)
        
        with open("api/text_detect.json", "w") as f:
            json.dump(detections, f, indent = 4)
        
        if plot:
            self._plot(save = True)
            
        return detections
        

if __name__ == "__main__":
    path = "api/modele-facture-fr-classique-blanc-750px.png"
    ocr = OCR(path, lang = ["fr"], gpu = True)
    ocr.detection()



        




