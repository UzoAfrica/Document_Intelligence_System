from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

class ModelLoader:
    def __init__(self):
        self.summarizer = pipeline("summarization", 
                                  model="facebook/bart-large-cnn")
        self.ner_pipeline = pipeline("ner", 
                                    model="dslim/bert-base-NER")
        self.classifier = pipeline("zero-shot-classification",
                                  model="facebook/bart-large-mnli")
        
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Generate summary of document"""
        return self.summarizer(text, max_length=max_length, 
                              min_length=30, do_sample=False)[0]['summary_text']
    
    def extract_entities(self, text: str) -> List[dict]:
        """Extract named entities"""
        return self.ner_pipeline(text)
    
    def classify_document(self, text: str, 
                          candidate_labels: List[str]) -> dict:
        """Classify document type"""
        return self.classifier(text[:512], candidate_labels)