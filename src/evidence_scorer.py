# medqa_project/src/evidence_scorer.py
import torch
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import nltk
import os
import json

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

class EvidenceScorer:
    """
    Scores answer options based on relevance (semantic + cross-encoder re-ranking),
    NLI, and Extractive Question Answering (QA) confidence.
    """

    def __init__(self, sentence_model: SentenceTransformer, device, batch_size: int = 32, nli_model_path: str = None,
                 cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", qa_model_path: str = None):
        """
        Initializes the EvidenceScorer.

        Parameters
        ----------
        sentence_model : SentenceTransformer
            The pre-loaded sentence transformer model (for initial retrieval).
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') the models are on.
        batch_size : int, optional
            Batch size for encoding evidence segments, by default 32.
        nli_model_path : str, optional
            Path to the fine-tuned NLI model directory. If None, NLI scoring will be skipped.
        cross_encoder_model_name : str, optional
            Name of the Cross-Encoder model to load from HuggingFace, by default "cross-encoder/ms-marco-MiniLM-L-6-v2".
        qa_model_path : str, optional
            Path to the fine-tuned QA model directory. If None, QA scoring will be skipped.
        """
        self.sentence_model = sentence_model
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()

        # Load NLI model and tokenizer
        self.nli_model = None
        self.nli_tokenizer = None
        self.nli_id_to_label = None
        self.nli_label_to_id = None

        if nli_model_path and os.path.exists(nli_model_path):
            try:
                logging.info(f"Loading NLI model from: {nli_model_path}")
                self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to(self.device)
                self.nli_model.eval()
                mappings_path = os.path.join(nli_model_path, "nli_label_mappings.json")
                if os.path.exists(mappings_path):
                    with open(mappings_path, "r") as f:
                        mappings = json.load(f)
                        self.nli_id_to_label = {int(k): v for k, v in mappings.get("nli_id_to_label", {}).items()}
                        self.nli_label_to_id = {v: int(k) for k, v in mappings.get("nli_label_to_id", {}).items()}
                    logging.info("NLI model and label mappings loaded successfully.")
                else:
                    logging.warning(f"nli_label_mappings.json not found in {nli_model_path}")
            except Exception as e:
                logging.error(f"Error loading NLI model from {nli_model_path}: {e}")
                self.nli_model = None
                self.nli_tokenizer = None
        else:
            logging.info("No NLI model path provided or path does not exist. NLI scoring will be skipped.")

        # Load Cross-Encoder model for re-ranking
        self.cross_encoder_tokenizer = None
        self.cross_encoder_model = None
        try:
            logging.info(f"Loading Cross-Encoder model: {cross_encoder_model_name}")
            self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)
            self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name).to(self.device)
            self.cross_encoder_model.eval()
            logging.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Cross-Encoder model {cross_encoder_model_name}: {e}")
            self.cross_encoder_model = None
            self.cross_encoder_tokenizer = None

        # Load QA model and tokenizer
        self.qa_model = None
        self.qa_tokenizer = None
        if qa_model_path and os.path.exists(qa_model_path):
            try:
                logging.info(f"Loading QA model from: {qa_model_path}")
                self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_path)
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path).to(self.device)
                self.qa_model.eval()
                logging.info("QA model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading QA model from {qa_model_path}: {e}")
                self.qa_model = None
                self.qa_tokenizer = None
        else:
            logging.info("No QA model path provided or path does not exist. QA scoring will be skipped.")

    def _segment_texts(self, page_contents: list) -> list:
        """
        Segments a list of text contents into individual sentences.
        """
        if not page_contents or not isinstance(page_contents, list):
            return []
        all_sentences = []
        for content in page_contents:
            if content and isinstance(content, str):
                try:
                    sentences = nltk.sent_tokenize(content)
                    all_sentences.extend([s.strip() for s in sentences if s.strip()])
                except Exception as e:
                    logging.warning(f"Could not tokenize content part due to: {e}. Snippet: {content[:100]}...")
        return all_sentences

    def _get_nli_prediction(self, premise: str, hypothesis: str):
        """
        Gets NLI prediction (label and confidence) for a premise-hypothesis pair.
        """
        if not self.nli_model or not self.nli_tokenizer:
            return None, None
        try:
            inputs = self.nli_tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
            probabilities = logits.softmax(dim=-1)
            confidence, predicted_class_id = probabilities.max(dim=-1)
            predicted_label = self.nli_id_to_label.get(predicted_class_id.item(), "UNKNOWN_NLI_LABEL") if self.nli_id_to_label else str(predicted_class_id.item())
            return predicted_label, confidence.item()
        except Exception as e:
            logging.error(f"Error during NLI prediction for premise '{premise[:50]}...' and hypothesis '{hypothesis[:50]}...': {e}")
            return "NLI_ERROR", 0.0

    def _re_rank_passages_with_cross_encoder(self, query: str, passages: list):
        """
        Re-ranks passages using a Cross-Encoder model.
        """
        if not self.cross_encoder_model or not self.cross_encoder_tokenizer or not passages:
            return [(p, 0.0) for p in passages]
        sentence_pairs = [[query, passage] for passage in passages]
        features = self.cross_encoder_tokenizer(
            sentence_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            scores = self.cross_encoder_model(**features).logits.squeeze(dim=-1)
        ranked_passages = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked_passages

    def _get_qa_prediction(self, question: str, context: str):
        """
        Gets QA prediction (answer span and confidence) for a question-context pair.
        Returns (predicted_answer_text, confidence_score).
        """
        if not self.qa_model or not self.qa_tokenizer:
            return "", 0.0
        try:
            inputs = self.qa_tokenizer(
                question, context,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
            answer_start_scores = start_logits.softmax(dim=-1)
            answer_end_scores = end_logits.softmax(dim=-1)
            answer_start = start_logits.argmax()
            answer_end = end_logits.argmax() + 1
            confidence = answer_start_scores[0, answer_start].item() * answer_end_scores[0, answer_end - 1].item()
            input_ids = inputs["input_ids"].squeeze().tolist()
            if answer_start >= len(input_ids) or answer_end > len(input_ids) or answer_start >= answer_end:
                return "", 0.0
            predicted_answer = self.qa_tokenizer.decode(input_ids[answer_start:answer_end], skip_special_tokens=True)
            return predicted_answer, confidence
        except Exception as e:
            logging.error(f"Error during QA prediction for question '{question[:50]}...' and context '{context[:50]}...': {e}")
            return "", 0.0

    def score_options(self, processed_question_data: dict, wikipedia_page_texts: list, top_k_initial_evidence: int = 20, top_k_reranked_evidence: int = 5, nli_weight: float = 1.0, qa_weight: float = 1.0, nli_confidence_threshold: float = 0.75, qa_confidence_threshold: float = 0.05) -> dict:
        """
        Scores answer options based on relevance (semantic + cross-encoder re-ranking),
        NLI, and Extractive Question Answering (QA) confidence.

        Parameters
        ----------
        processed_question_data : dict
            Processed question data including original question and options.
        wikipedia_page_texts : list
            List of text content from selected Wikipedia pages.
        top_k_initial_evidence : int, optional
            Number of top similar evidence segments to retrieve initially via FAISS, by default 20.
        top_k_reranked_evidence : int, optional
            Number of top evidence segments to re-rank with Cross-Encoder and pass to NLI/QA, by default 5.
        nli_weight : float, optional
            Weight to give to the NLI score relative to cross-encoder score, by default 1.0.
        qa_weight : float, optional
            Weight to give to the QA confidence score, by default 1.0.
        nli_confidence_threshold : float, optional
            Minimum confidence for an NLI prediction to be considered, by default 0.75.
        qa_confidence_threshold : float, optional
            Minimum confidence for a QA prediction to be considered, by default 0.05.

        Returns
        -------
        dict
            A dictionary mapping option keys to their final calculated scores.
        """
        original_question_text = processed_question_data.get('original_question_data', {}).get('question')
        options = processed_question_data.get('original_question_data', {}).get('options')

        if not original_question_text or not options or not isinstance(options, dict) or not wikipedia_page_texts:
            logging.warning("Insufficient data for scoring.")
            return {}

        logging.info(f"  EvidenceScorer: Segmenting {len(wikipedia_page_texts)} Wikipedia pages...")
        evidence_segments = self._segment_texts(wikipedia_page_texts)
        if not evidence_segments:
            logging.info("  EvidenceScorer: No evidence segments found.")
            return {key: 0.0 for key in options.keys()}

        logging.info(f"  EvidenceScorer: Encoding {len(evidence_segments)} evidence segments for initial retrieval...")
        try:
            segment_embeddings_np = self.sentence_model.encode(
                evidence_segments,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            faiss.normalize_L2(segment_embeddings_np)
        except Exception as e:
            logging.error(f"  EvidenceScorer: Error encoding evidence segments: {e}")
            return {key: 0.0 for key in options.keys()}

        try:
            index = faiss.IndexFlatIP(self.embedding_dim)
            if self.device.type == 'cuda' and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources();
                index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(segment_embeddings_np.astype(np.float32))
            logging.info(f"  EvidenceScorer: FAISS index built with {index.ntotal} vectors.")
        except Exception as e:
            logging.error(f"  EvidenceScorer: Error building FAISS index: {e}")
            return {key: 0.0 for key in options.keys()}

        option_final_scores = {}
        for option_key, option_text in options.items():
            if not option_text or not isinstance(option_text, str):
                option_final_scores[option_key] = 0.0
                continue

            # Create the query for relevance scoring and hypothesis for NLI/QA
            # For QA, the question should be the original question text.
            # We will then check if the extracted answer span matches the option.
            qa_question = original_question_text
            # For Cross-Encoder and NLI, the combined text remains effective
            combined_query_text = f"{original_question_text} {option_text}"

            max_option_score = -100.0 # Start with a very low score for cross-encoder logits

            try:
                # 1. Initial Retrieval (Semantic Similarity with SentenceTransformer)
                query_embedding_np = self.sentence_model.encode(combined_query_text, convert_to_numpy=True)
                query_embedding_normalized = query_embedding_np.reshape(1, -1)
                faiss.normalize_L2(query_embedding_normalized)

                distances, indices = index.search(query_embedding_normalized.astype(np.float32), k=top_k_initial_evidence)

                retrieved_passages = []
                if distances.size > 0 and indices[0][0] != -1:
                    for idx in indices[0]:
                        if idx != -1:
                            retrieved_passages.append(evidence_segments[idx])

                if not retrieved_passages:
                    logging.info(f"    Option '{option_key}': No relevant evidence found in initial FAISS retrieval.")
                    option_final_scores[option_key] = 0.0
                    continue

                # 2. Re-ranking with Cross-Encoder
                logging.info(f"    Option '{option_key}': Re-ranking {len(retrieved_passages)} passages with Cross-Encoder...")
                ranked_passages_with_scores = self._re_rank_passages_with_cross_encoder(combined_query_text, retrieved_passages)

                # Take top_k_reranked_evidence for NLI, QA, and final scoring
                top_reranked_passages = ranked_passages_with_scores[:top_k_reranked_evidence]

                if not top_reranked_passages:
                    logging.info(f"    Option '{option_key}': No passages left after Cross-Encoder re-ranking.")
                    option_final_scores[option_key] = 0.0
                    continue

                # Initialize scores for aggregation
                nli_contributions = [] # Store positive/negative NLI confidences
                qa_confidences_for_option = [] # Store QA prediction confidences if they match option
                cross_encoder_base_scores = [] # Store CE scores of passages that contribute to NLI/QA

                for passage_text, ce_score in top_reranked_passages:
                    # NLI Scoring
                    if self.nli_model:
                        nli_label, nli_confidence = self._get_nli_prediction(passage_text, combined_query_text)
                        if nli_confidence is not None and nli_confidence >= nli_confidence_threshold:
                            if nli_label == "ENTAILMENT":
                                nli_contributions.append(nli_confidence)
                                cross_encoder_base_scores.append(ce_score)
                            elif nli_label == "CONTRADICTION":
                                nli_contributions.append(-nli_confidence)
                                cross_encoder_base_scores.append(ce_score)

                    # QA Scoring
                    if self.qa_model:
                        predicted_answer_span, qa_confidence = self._get_qa_prediction(qa_question, passage_text)
                        # Check if the predicted answer span contains the option text or is very similar
                        # Use a more robust comparison: check if option words are in predicted span, or vice versa
                        option_words = set(word.lower() for word in option_text.split() if word.lower() not in nltk.corpus.stopwords.words('english'))
                        predicted_span_words = set(word.lower() for word in predicted_answer_span.split() if word.lower() not in nltk.corpus.stopwords.words('english'))

                        # Consider a match if there's significant overlap or one contains the other
                        match_found = False
                        if option_words and predicted_span_words:
                            overlap = len(option_words.intersection(predicted_span_words))
                            if overlap > 0: # At least one common word
                                # Stronger match if one is contained in the other, or high overlap ratio
                                if option_text.lower() in predicted_answer_span.lower() or \
                                   predicted_answer_span.lower() in option_text.lower() or \
                                   (overlap / len(option_words) > 0.5 and overlap / len(predicted_span_words) > 0.5):
                                    match_found = True

                        if match_found and qa_confidence >= qa_confidence_threshold:
                            qa_confidences_for_option.append(qa_confidence)
                            if ce_score not in cross_encoder_base_scores: # Avoid duplicate CE scores if NLI also used this passage
                                cross_encoder_base_scores.append(ce_score)

                # 4. Combine Scores for Final Option Score
                final_score_components = []

                # Add aggregated Cross-Encoder score
                if cross_encoder_base_scores:
                    final_score_components.append(max(cross_encoder_base_scores))
                else:
                    # If no passages met NLI/QA criteria, take the max CE score from top_reranked_passages
                    # This ensures a base relevance score even if NLI/QA don't fire
                    if top_reranked_passages:
                        final_score_components.append(max([s for _, s in top_reranked_passages]))
                    else:
                        final_score_components.append(0.0) # No relevant passages found

                # Add aggregated NLI score
                if nli_contributions and self.nli_model:
                    aggregated_nli_score = 0.0
                    positive_nli_scores = [s for s in nli_contributions if s > 0]
                    negative_nli_scores = [s for s in nli_contributions if s < 0]

                    if positive_nli_scores:
                        aggregated_nli_score = max(positive_nli_scores)
                    elif negative_nli_scores:
                        aggregated_nli_score = min(negative_nli_scores)

                    final_score_components.append(nli_weight * aggregated_nli_score)

                # Add aggregated QA score
                if qa_confidences_for_option and self.qa_model:
                    final_score_components.append(qa_weight * max(qa_confidences_for_option)) # Take max QA confidence

                # Final score is the sum of relevant components
                if final_score_components:
                    max_option_score = sum(final_score_components)
                else:
                    max_option_score = 0.0 

            except Exception as e:
                logging.error(f"  EvidenceScorer: Error during processing for option '{option_key}': {e}")
                max_option_score = 0.0

            option_final_scores[option_key] = max_option_score

        logging.info(f"  EvidenceScorer: Final scores for options: {option_final_scores}")
        return option_final_scores

