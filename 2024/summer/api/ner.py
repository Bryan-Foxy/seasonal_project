import json 
import spacy

class NER:
    def __init__(self, path, lang = "fr", entities = []):
        super(NER, self).__init__()
        self.path = path
        self.entities = entities
        if lang == "fr":
            self.nlp = spacy.load('fr_core_news_sm')
        else:
            self.nlp = spacy.load('en_core_sm')
    
    def _load_data(self):
        with open("api/text_detect.json", "r") as f:
            data = json.load(f)
        return data 
    
    def detection(self):
        data = self._load_data()
        doc = self.nlp(data)





        

