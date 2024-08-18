import json 
import spacy
from tqdm import tqdm 

class NER:
    def __init__(self, path, lang = "fr", entities = []):
        super(NER, self).__init__()
        self.path = path
        self.entities = entities
        if lang == "fr":
            self.nlp = spacy.load('fr_core_news_sm')
        else:
            self.nlp = spacy.load('en_core_web_sm')
    
    def _load_data(self):
        with open(self.path, "r") as f:
            data = json.load(f)
        text_data = " ".join(item['text'] for item in data)
        return text_data 
    
    def detection(self):
        data = self._load_data()
        doc = self.nlp(data)

        # Dict to store result of the NER
        extracted_entities = {
            "names": [],
            "prices": [],
            "dates": [],
            "organizations": []
        }
        for ent in tqdm(doc.ents):
            if ent.label_ == "PER":
                extracted_entities["names"].append(ent.text)
            elif ent.label_ == "ORG":
                extracted_entities["organizations"].append(ent.text)
            elif ent.label_ == "DATE":
                extracted_entities["dates"].append(ent.text)
            elif ent.label == ["MONEY","NUM"]:
                extracted_entities["prices"].append(ent.text)
        
        print("Noms:", extracted_entities["names"])
        print("Organisations:", extracted_entities["organizations"])
        print("Date:", extracted_entities["dates"])
        print("Prix:", extracted_entities["prices"])


if __name__ == "__main__":
    path = "api/text_detect.json"
    ner = NER(path)
    ner.detection()




        

