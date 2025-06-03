# medqa_project/src/evidence_scorer.py
import torch
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer
import nltk
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification # For NLI
import os # For NLI model path
import json # For NLI label mappings
import traceback

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)

class EvidenceScorer:
    """
    Scores answer options based on similarity and/or NLI with Wikipedia segments.
    """

    def __init__(self, sentence_model: SentenceTransformer, device: torch.device, 
                 batch_size: int = 32, nli_model_path: str = None):
        self.sentence_model = sentence_model
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()

        # Load NLI model and tokenizer if path is provided
        self.nli_model = None
        self.nli_tokenizer = None
        self.nli_id_to_label = None # To interpret NLI output
        if nli_model_path and os.path.exists(nli_model_path):
            try:
                print(f"Loading NLI model from: {nli_model_path}")
                self.nli_tokenizer = DistilBertTokenizerFast.from_pretrained(nli_model_path)
                self.nli_model = DistilBertForSequenceClassification.from_pretrained(nli_model_path).to(self.device)
                self.nli_model.eval() # Set to evaluation mode
                
                mappings_path = os.path.join(nli_model_path, "nli_label_mappings.json")
                if os.path.exists(mappings_path):
                    with open(mappings_path, "r") as f:
                        mappings = json.load(f)
                        self.nli_id_to_label = {int(k): v for k,v in mappings.get("nli_id_to_label", {}).items()}
                    print("NLI model and label mappings loaded successfully.")
                else:
                    print(f"Warning: nli_label_mappings.json not found in {nli_model_path}")

            except Exception as e:
                print(f"Error loading NLI model from {nli_model_path}: {e}")
                self.nli_model = None
                self.nli_tokenizer = None
        else:
            print("No NLI model path provided or path does not exist. NLI scoring will be skipped.")


    def _segment_texts(self, page_contents: list) -> list:
        if not page_contents or not isinstance(page_contents, list): return []
        all_sentences = []
        for content in page_contents:
            if content and isinstance(content, str):
                try:
                    sentences = nltk.sent_tokenize(content)
                    all_sentences.extend([s.strip() for s in sentences if s.strip()])
                except Exception as e:
                    print(f"Warning: Could not tokenize content part due to: {e}. Snippet: {content[:100]}...")
        return all_sentences

    def _get_nli_score(self, premise: str, hypothesis: str) -> tuple[str | None, float | None]:
        """Gets NLI prediction and confidence for a premise-hypothesis pair."""
        if not self.nli_model or not self.nli_tokenizer:
            return None, None
        
        try:
            inputs = self.nli_tokenizer(
                premise, hypothesis, 
                return_tensors="pt", 
                truncation=True, # Truncate premise+hypothesis pair
                max_length=256, # Should match NLI model's training
                padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
            
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class_id = torch.max(probabilities, dim=-1)
            
            predicted_label = None
            if self.nli_id_to_label:
                predicted_label = self.nli_id_to_label.get(predicted_class_id.item(), "UNKNOWN_NLI_LABEL")
            else:
                predicted_label = str(predicted_class_id.item())

            return predicted_label, confidence.item()

        except Exception as e:
            print(f"Error during NLI prediction for premise '{premise[:50]}...' and hypothesis '{hypothesis[:50]}...': {e}")
            return "NLI_ERROR", 0.0


    def score_options(self, processed_question_data: dict, wikipedia_page_texts: list, top_k_evidence: int = 3) -> dict:
        original_question_text = processed_question_data.get('original_question_data', {}).get('question')
        options = processed_question_data.get('original_question_data', {}).get('options')

        if not original_question_text or not options or not isinstance(options, dict) or not wikipedia_page_texts:
            print("Warning: Insufficient data for scoring.")
            return {}

        print(f"  EvidenceScorer: Segmenting {len(wikipedia_page_texts)} Wikipedia pages...")
        evidence_segments = self._segment_texts(wikipedia_page_texts)
        if not evidence_segments:
            print("  EvidenceScorer: No evidence segments found.")
            return {key: 0.0 for key in options.keys()}

        print(f"  EvidenceScorer: Encoding {len(evidence_segments)} evidence segments...")
        try:
            segment_embeddings_np = self.sentence_model.encode(evidence_segments, batch_size=self.batch_size, convert_to_numpy=True)
            faiss.normalize_L2(segment_embeddings_np)
        except Exception as e:
            print(f"  EvidenceScorer: Error encoding evidence segments: {e}")
            return {key: 0.0 for key in options.keys()}
        
        try:
            index = faiss.IndexFlatIP(self.embedding_dim)
            if self.device.type == 'cuda' and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources(); index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(segment_embeddings_np.astype(np.float32))
            print(f"  EvidenceScorer: FAISS index built with {index.ntotal} vectors.")
        except Exception as e:
            print(f"  EvidenceScorer: Error building FAISS index: {e}")
            return {key: 0.0 for key in options.keys()}

        option_final_scores = {}
        for option_key, option_text in options.items():
            if not option_text or not isinstance(option_text, str):
                option_final_scores[option_key] = 0.0; continue

            query_text = f"{original_question_text} <SEP> {option_text}"
            max_option_score = -1.0  # Using -1 as initial for similarity scores that can be negative

            try:
                query_embedding_np = self.sentence_model.encode(query_text, convert_to_numpy=True)
                query_embedding_normalized = query_embedding_np.reshape(1, -1)
                faiss.normalize_L2(query_embedding_normalized)
            except Exception as e:
                print(f"  EvidenceScorer: Error encoding query for option '{option_key}': {e}")
                option_final_scores[option_key] = 0.0; continue
            
            try:
                # Retrieve top_k_evidence for NLI or for max similarity
                distances, indices = index.search(query_embedding_normalized.astype(np.float32), k=top_k_evidence)
                
                if distances.size > 0 and indices[0][0] != -1:
                    # Use similarity score as a base
                    base_similarity_score = float(distances[0][0]) # Highest similarity
                    max_option_score = base_similarity_score
                    
                    # NLI scoring (if model available)
                    if self.nli_model:
                        nli_weighted_score = 0.0
                        num_nli_scores = 0
                        print(f"    NLI for Option '{option_key}':")
                        for i in range(len(indices[0])):
                            if indices[0][i] == -1: continue # Skip invalid index
                            
                            premise_text = evidence_segments[indices[0][i]]
                            nli_label, nli_confidence = self._get_nli_score(premise_text, query_text) # query_text is hypothesis
                            print(f"      Premise (sim: {distances[0][i]:.2f}): '{premise_text[:80]}...' -> NLI: {nli_label} (Conf: {nli_confidence:.2f})")

                            if nli_label == "ENTAILMENT":
                                nli_weighted_score += (1.0 * nli_confidence) # Strong positive
                            elif nli_label == "CONTRADICTION":
                                nli_weighted_score -= (1.0 * nli_confidence) # Strong negative
                            # Neutral might slightly reduce score or be ignored
                            # elif nli_label == "NEUTRAL":
                            #     nli_weighted_score += (0.0 * nli_confidence)
                            num_nli_scores+=1
                        
                        # Combine NLI scores (e.g., average or max, and then combine with similarity)
                        # This is a simple combination, can be more sophisticated
                        if num_nli_scores > 0:
                             # Example: NLI score contributes significantly
                            final_score_for_option = base_similarity_score + (nli_weighted_score / num_nli_scores if num_nli_scores > 0 else 0)
                            max_option_score = final_score_for_option 
                        # If NLI is not decisive, rely on similarity
                    
                else: # No relevant evidence from similarity search
                    max_option_score = 0.0 
                    print(f"    Option '{option_key}': No relevant evidence found in FAISS for similarity.")


            except Exception as e:
                print(f"  EvidenceScorer: Error during FAISS search or NLI for option '{option_key}': {e}")
                max_option_score = 0.0
            
            option_final_scores[option_key] = max_option_score
        
        print(f"  EvidenceScorer: Final scores for options: {option_final_scores}")
        return option_final_scores


if __name__ == '__main__':
    from src.question_processor import QuestionProcessor 
    
    example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_nli_model_path = "../../models/nli_classifier" # Path where dummy NLI model might be saved

    if not os.path.exists(dummy_nli_model_path):
        print(f"Warning: NLI model not found at '{dummy_nli_model_path}'. NLI scoring will be skipped in example.")
        print("Run 'python src/fine_tuning/nli_tuner.py' to create a dummy model.")
        dummy_nli_model_path = None # Ensure NLI is skipped if model not found

    try:
        q_processor = QuestionProcessor(device=example_device)
        # Initialize with NLI model path if available
        scorer = EvidenceScorer(sentence_model=q_processor.sentence_model, 
                                device=example_device,
                                nli_model_path=dummy_nli_model_path)

        sample_processed_question = {
            "original_question_data": {
                "question": "What is the primary treatment for type 1 diabetes?",
                "options": {"A": "Oral hypoglycemic agents", "B": "Insulin therapy"},
                "answer_idx": "B"
            },
        }
        sample_wiki_texts = [
            "Type 1 diabetes treatment focuses on managing blood sugar levels with insulin, diet and lifestyle.",
            "Insulin therapy is the cornerstone of treatment for type 1 diabetes.",
            "Oral hypoglycemic agents are not effective for type 1 diabetes."
        ]
        
        option_scores = scorer.score_options(sample_processed_question, sample_wiki_texts, top_k_evidence=2)
        print("\n--- Option Scores (with NLI if model loaded) ---")
        for option, score in option_scores.items(): print(f"Option {option}: {score:.4f}")

    except Exception as e:
        print(f"An error occurred during EvidenceScorer NLI example usage: {e}")
        traceback.print_exc()
