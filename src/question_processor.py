# medqa_project/src/question_processor.py
import spacy
from sentence_transformers import SentenceTransformer
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import json
import logging

class QuestionProcessor:
    """
    Processes individual questions: NER, sentence embeddings, and optional question classification.
    """

    def __init__(self, device, classifier_model_path: str = None):
        """
        Initializes the QuestionProcessor.

        Parameters
        ----------
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') for models.
        classifier_model_path : str, optional
            Path to the fine-tuned question classifier model directory.
            If None, classification will be skipped.
        """
        self.device = device
        try:
            self.nlp_spacy = spacy.load("en_core_sci_sm", disable=['parser', 'tagger'])
            logging.info("SciSpacy model 'en_core_sci_sm' loaded successfully.")
        except OSError:
            logging.error("SciSpacy model 'en_core_sci_sm' not found. Please run: python -m spacy download en_core_sci_sm")
            raise

        try:
            self.sentence_model = SentenceTransformer("all-mpnet-base-v2").to(self.device)
            logging.info(f"SentenceTransformer model 'all-mpnet-base-v2' loaded successfully on {self.device}.")
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer model: {e}")
            raise

        self.classifier_model = None
        self.classifier_tokenizer = None
        self.id_to_label_map = None
        if classifier_model_path and os.path.exists(classifier_model_path):
            try:
                logging.info(f"Loading question classifier from: {classifier_model_path}")
                self.classifier_tokenizer = DistilBertTokenizerFast.from_pretrained(classifier_model_path)
                self.classifier_model = DistilBertForSequenceClassification.from_pretrained(classifier_model_path).to(self.device)
                self.classifier_model.eval()
                mappings_path = os.path.join(classifier_model_path, "label_mappings.json")
                if os.path.exists(mappings_path):
                    with open(mappings_path, "r") as f:
                        mappings = json.load(f)
                        self.id_to_label_map = {int(k): v for k, v in mappings.get("id_to_label", {}).items()}
                    logging.info("Question classifier and label mappings loaded successfully.")
                else:
                    logging.warning(f"label_mappings.json not found in {classifier_model_path}. Classification might output IDs.")
            except Exception as e:
                logging.error(f"Error loading question classifier model from {classifier_model_path}: {e}")
                self.classifier_model = None
                self.classifier_tokenizer = None
                self.id_to_label_map = None
        else:
            logging.info("No classifier model path provided or path does not exist. Classification will be skipped.")

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return text.lower().strip()

    def _classify_question(self, text: str) -> str | None:
        """Classifies the question text if a classifier model is loaded."""
        if not self.classifier_model or not self.classifier_tokenizer or not text:
            return None
        try:
            inputs = self.classifier_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            with torch.no_grad():
                logits = self.classifier_model(**inputs).logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            if self.id_to_label_map:
                return self.id_to_label_map.get(predicted_class_id, "UNKNOWN_CATEGORY")
            else:
                return str(predicted_class_id)
        except Exception as e:
            logging.error(f"Error during question classification: {e}")
            return "CLASSIFICATION_ERROR"

    def process(self, question_data: dict) -> dict:
        if not question_data:
            return {
                "original_question_data": {}, "cleaned_question_text": "",
                "extracted_entities": [], "question_embedding": None,
                "question_category": None
            }
        metamap_entities = set(self._clean_text(phrase) for phrase in question_data.get("metamap_phrases", []) if phrase)
        ner_entities = set()
        question_text = question_data.get("question", "")
        if question_text and self.nlp_spacy:
            doc = self.nlp_spacy(question_text)
            for ent in doc.ents:
                ner_entities.add(self._clean_text(ent.text))
        options = question_data.get("options", {})
        if isinstance(options, dict) and self.nlp_spacy:
            for option_text in options.values():
                if option_text:
                    doc = self.nlp_spacy(option_text)
                    for ent in doc.ents:
                        ner_entities.add(self._clean_text(ent.text))
        all_entities = list(metamap_entities.union(ner_entities))
        cleaned_question = self._clean_text(question_text)
        question_embedding = None
        if cleaned_question and self.sentence_model:
            embedding_np = self.sentence_model.encode(cleaned_question)
            question_embedding = torch.from_numpy(embedding_np).to(self.device)
        question_category = self._classify_question(cleaned_question)
        return {
            "original_question_data": question_data,
            "cleaned_question_text": cleaned_question,
            "extracted_entities": all_entities,
            "question_embedding": question_embedding,
            "question_category": question_category
        }
