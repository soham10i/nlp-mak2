# medqa_project/src/page_selector.py

import wikipedia
import requests
from bs4 import BeautifulSoup
import time
import torch
from sentence_transformers import SentenceTransformer
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import logging

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.warning("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    logging.warning("NLTK 'stopwords' corpus not found. Downloading...")
    nltk.download('stopwords', quiet=True)

class PageSelector:
    """
    Selects relevant Wikipedia pages for a given processed question.
    """

    def __init__(self, sentence_model: SentenceTransformer, device: torch.device, batch_size: int = 128):
        """
        Initializes the PageSelector.

        Parameters
        ----------
        sentence_model : SentenceTransformer
            The pre-loaded sentence transformer model.
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') the model is on.
        batch_size : int, optional
            Batch size for encoding summaries, by default 128.
        """
        self.sentence_model = sentence_model
        self.device = device
        self.batch_size = batch_size
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        logging.info("PageSelector initialized.")

    def _fetch_page_content_requests(self, title: str) -> str:
        """
        Fetches the full plain text content of a Wikipedia page using requests and BeautifulSoup,
        with a fallback to wikipedia.page().content if parsing fails or specific elements are not found.
        """
        headers = {
            'User-Agent': 'MedQAProject/1.0 (your-contact-email@example.com; http://your-project-url.com)'
        }
        content_from_requests = ""
        try:
            page = wikipedia.page(title=title, auto_suggest=False, redirect=True)
            url = page.url
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div:
                content_div = soup.find('div', id='bodyContent')
            if content_div:
                for unwanted_tag in content_div.find_all(['table', 'div.infobox', 'div.navbox', 'span.mw-editsection', 'div.thumb', 'div.noprint', 'div.hatnote', 'sup.reference', 'ol.references']):
                    unwanted_tag.decompose()
                texts = [p.get_text(separator=' ', strip=True) for p in content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])]
                content_from_requests = "\n".join(filter(None, texts)).strip()
                if content_from_requests:
                    return content_from_requests
                else:
                    logging.warning(f"Requests/BeautifulSoup parsing resulted in empty content for '{title}'. Trying wikipedia.page().content.")
        except wikipedia.exceptions.PageError:
            logging.error(f"Wikipedia page '{title}' not found (PageError).")
            return ""
        except wikipedia.exceptions.DisambiguationError as e:
            logging.error(f"'{title}' is a disambiguation page. Options: {e.options[:5]}. Returning empty content.")
            return ""
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching page '{title}' with requests: {e}. Trying wikipedia.page().content as fallback.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during requests/BeautifulSoup for '{title}': {e}. Trying wikipedia.page().content as fallback.")
        try:
            page = wikipedia.page(title=title, auto_suggest=False, redirect=True)
            content_from_wiki_lib = page.content.strip()
            if content_from_wiki_lib:
                logging.info(f"Successfully fetched content for '{title}' using wikipedia.page().content.")
                return content_from_wiki_lib
            else:
                logging.warning(f"wikipedia.page().content also returned empty for '{title}'.")
        except wikipedia.exceptions.PageError:
            logging.error(f"Wikipedia page '{title}' not found on fallback.")
        except wikipedia.exceptions.DisambiguationError as e:
            logging.error(f"'{title}' is a disambiguation page on fallback. Options: {e.options[:5]}.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during wikipedia.page().content fallback for '{title}': {e}")
        return ""

    def select_pages(self, processed_question_data: dict, max_pages: int = 5, max_candidates_per_query: int = 20, bm25_weight: float = 0.25) -> list:
        """
        Selects up to max_pages relevant Wikipedia pages for the processed question.

        Parameters
        ----------
        processed_question_data : dict
            The output from QuestionProcessor.process(), containing at least:
            - 'extracted_entities': list of strings
            - 'question_embedding': torch.Tensor
            - 'cleaned_question_text': str (for combined search)
            - 'question_category': str (for category-aware query expansion)
        max_pages : int, optional
            The maximum number of Wikipedia pages to return, by default 5.
        max_candidates_per_query : int, optional
            Max Wikipedia search results to consider per entity query, by default 20.
        bm25_weight : float, optional
            Weight to give to the BM25 score when combining with semantic similarity, by default 0.25.

        Returns
        -------
        list
            A list of strings, where each string is the cleaned text content of a selected Wikipedia page.
        """
        extracted_entities = processed_question_data.get('extracted_entities', [])
        question_embedding = processed_question_data.get('question_embedding')
        cleaned_question_text = processed_question_data.get('cleaned_question_text', '')
        question_category = processed_question_data.get('question_category') # Get question category
        options_texts = list(processed_question_data.get('original_question_data', {}).get('options', {}).values())


        if not extracted_entities or question_embedding is None:
            logging.warning("No entities or question embedding provided to PageSelector.")
            return []

        # Stores {page_title: {'semantic_score': float, 'bm25_score': float, 'summary': str}}
        candidate_pages_detailed_info = {} 

        # --- Advanced Search Query Generation ---
        # 1. Filter out very short or common words from extracted entities
        #    Also, prioritize entities that are likely to be strong medical terms
        #    Expanded stop words list to include common medical measurement/time units
        medical_stop_words = self.stop_words.union([
            "man", "woman", "patient", "history", "exam", "shows", "due", "rate", "blood", 
            "mg", "mm", "hg", "c", "f", "ml", "dl", "ng", "min", "hr", "year", "old", "years",
            "day", "days", "week", "weeks", "month", "months", "level", "levels", "value", "values",
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "of", "in", "on", "at", "for", "with", "as", "by", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
            "which", "what", "how", "when", "where", "why", "who", "whom", "whose", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
        ])
        
        filtered_entities = []
        for entity in extracted_entities:
            # Keep multi-word entities or single words that are not common/medical stop words and are longer than 2 chars
            # Prioritize entities with multiple words or that are longer and not common.
            words = entity.split()
            if len(words) > 1 and all(word.lower() not in medical_stop_words for word in words):
                filtered_entities.append(entity)
            elif len(words) == 1 and len(words[0]) > 3 and words[0].lower() not in medical_stop_words:
                filtered_entities.append(entity)
        
        # Sort by length to prioritize longer, potentially more specific phrases
        filtered_entities = sorted(list(set(filtered_entities)), key=len, reverse=True)
        
        # 2. Add question category-aware query expansion
        category_keywords = {
            "DEFINITION": ["definition", "overview", "what is", "describe"],
            "TREATMENT_PROCEDURE": ["treatment", "therapy", "management", "procedure", "drug", "medication", "interventions"],
            "DIAGNOSIS_SYMPTOM": ["diagnosis", "symptoms", "signs", "clinical presentation", "diagnostic test", "manifestations"],
            "CAUSE_MECHANISM": ["cause", "etiology", "pathophysiology", "mechanism of action", "risk factors", "leads to"],
            "PROGNOSIS_OUTCOME": ["prognosis", "outcome", "survival rate", "complications", "recovery"],
            "PHARMACOLOGY_DRUG": ["pharmacology", "drug", "side effects", "dosage", "contraindications", "drug class"],
            "ANATOMY_PHYSIOLOGY": ["anatomy", "physiology", "function", "structure", "process", "role"],
            "EPIDEMIOLOGY_STATISTICS": ["epidemiology", "prevalence", "incidence", "statistics", "mortality", "demographic"],
            "PATHOLOGY_HISTOLOGY": ["pathology", "histology", "cellular basis", "changes", "metastasize"],
            "OTHER_GENERAL": [] 
        }
        
        additional_queries = []
        if question_category and question_category in category_keywords:
            for entity in filtered_entities[:5]: # Take top few filtered entities
                for kw in category_keywords[question_category]:
                    additional_queries.append(f"{entity} {kw}")
            # Also add general category-specific queries if the question is long enough
            if len(cleaned_question_text.split()) > 5:
                key_terms = [word for word in cleaned_question_text.split() if word.lower() not in medical_stop_words][:3]
                if key_terms:
                    additional_queries.append(f"{' '.join(key_terms)} {question_category.lower().replace('_', ' ')}") # Replace underscore for better search


        # 3. Option-based query generation (New and more aggressive)
        option_search_queries = []
        question_key_words = [word for word in cleaned_question_text.split() if word.lower() not in medical_stop_words and len(word) > 2]
        
        for option_text in options_texts:
            option_words = [word for word in option_text.lower().split() if word not in medical_stop_words and len(word) > 2]
            
            if option_words:
                # Query 1: Just the option's key words
                option_search_queries.append(" ".join(option_words))
                
                # Query 2: Option's key words + top question key words
                if question_key_words:
                    combined_option_query = " ".join(list(set(question_key_words[:3] + option_words[:3])))
                    if combined_option_query:
                        option_search_queries.append(combined_option_query)
                
                # Query 3: Option's key words + top filtered entities
                if filtered_entities:
                    combined_option_entity_query = " ".join(list(set(filtered_entities[:3] + option_words[:3])))
                    if combined_option_entity_query:
                        option_search_queries.append(combined_option_entity_query)


        # 4. Combine all search queries
        all_search_queries = []
        MAX_WIKI_QUERY_LENGTH = 300 
        
        # Prioritize a highly focused query from question + top entities
        focused_combined_query_parts = [word for word in cleaned_question_text.split() if word.lower() not in medical_stop_words][:5]
        if filtered_entities:
            focused_combined_query_parts.extend(filtered_entities[:3])
        focused_combined_query = " ".join(list(set(focused_combined_query_parts)))
        if focused_combined_query:
            all_search_queries.append(focused_combined_query[:MAX_WIKI_QUERY_LENGTH])
        
        # Add filtered entities, category-aware, and option-based queries
        all_search_queries.extend(filtered_entities)
        all_search_queries.extend(additional_queries)
        all_search_queries.extend(option_search_queries) # Use the new, more aggressive option queries
        
        all_search_queries = list(set(all_search_queries)) # Remove duplicates
        
        # Limit total search queries to avoid excessive API calls
        all_search_queries = all_search_queries[:50] # Increased total queries to try even more

        logging.info(f"Generated search queries: {all_search_queries}")

        for query in all_search_queries:
            if not query.strip(): 
                continue
            
            truncated_query = query[:MAX_WIKI_QUERY_LENGTH]

            try:
                logging.info(f"Searching Wikipedia for: '{truncated_query}'")
                search_results = wikipedia.search(truncated_query, results=max_candidates_per_query)
                time.sleep(0.2) 

                for page_title in search_results:
                    if page_title not in candidate_pages_detailed_info: # Only add if not already considered
                        try:
                            summary = wikipedia.summary(page_title, sentences=3, auto_suggest=False, redirect=True)
                            time.sleep(0.2) 
                            if summary:
                                candidate_pages_detailed_info[page_title] = {'summary': summary}
                                logging.info(f"Candidate found: '{page_title}'")
                        except wikipedia.exceptions.DisambiguationError:
                            logging.warning(f"Skipping disambiguation page: '{page_title}' during summary fetch.")
                        except wikipedia.exceptions.PageError:
                            logging.warning(f"Skipping page (not found for summary): '{page_title}'.")
                        except Exception as e:
                            logging.error(f"Error processing summary for '{page_title}': {e}")
            except Exception as e:
                logging.error(f"Error during Wikipedia search for '{truncated_query}': {e}")
        
        if not candidate_pages_detailed_info:
            logging.info("No candidate pages found after initial searches.")
            return []

        # --- Hybrid Ranking (Semantic + BM25) ---
        corpus = [info['summary'] for info in candidate_pages_detailed_info.values()]
        titles = list(candidate_pages_detailed_info.keys())
        
        # Tokenize corpus for BM25
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus] # Ensure lowercasing for BM25
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize question for BM25
        tokenized_question = cleaned_question_text.lower().split(" ") # Ensure lowercasing for BM25

        # Calculate semantic embeddings for all summaries in one go
        summary_embeddings_np = self.sentence_model.encode(corpus, batch_size=self.batch_size, convert_to_numpy=True)
        summary_embeddings = torch.from_numpy(summary_embeddings_np).to(self.device)

        # Ensure question embedding is 2D for cosine_similarity
        q_emb = question_embedding.unsqueeze(0) if len(question_embedding.shape) == 1 else question_embedding

        final_ranked_candidates = {}
        
        # Get all BM25 scores for the query against the corpus
        bm25_scores_all = bm25.get_scores(tokenized_question) 
        max_bm25_score = max(bm25_scores_all) if bm25_scores_all.size > 0 else 1.0 # Avoid division by zero

        # Define a small positive offset to ensure scores are generally positive for relevant pages
        # This helps prevent all scores from being negative and makes ranking more intuitive.
        # This offset should be carefully tuned; 0.5 is a starting point if scores are typically around -1 to 1.
        POSITIVE_SCORE_OFFSET = 0.5 

        for i, title in enumerate(titles):
            summary = candidate_pages_detailed_info[title]['summary']

            # Calculate Semantic Similarity
            s_emb = summary_embeddings[i].unsqueeze(0) 
            semantic_score = torch.nn.functional.cosine_similarity(q_emb, s_emb).item()

            # Get BM25 Score for the current document and Normalize
            bm25_score = bm25_scores_all[i] 
            normalized_bm25_score = bm25_score / (max_bm25_score + 1e-6) # Min-max normalization for BM25

            # Normalize semantic score to be between 0 and 1
            normalized_semantic_score = (semantic_score + 1) / 2 # Scale from [-1, 1] to [0, 1]
            
            # Final combined score with positive offset
            combined_score = (1 - bm25_weight) * normalized_semantic_score + bm25_weight * normalized_bm25_score + POSITIVE_SCORE_OFFSET
            
            final_ranked_candidates[title] = combined_score
            logging.info(f"Ranked Candidate: '{title}', Semantic: {semantic_score:.4f}, BM25 (Norm): {normalized_bm25_score:.4f}, Combined: {combined_score:.4f}")

        # Sort all collected candidates by their combined score
        sorted_candidates = sorted(final_ranked_candidates.items(), key=lambda item: item[1], reverse=True)
        
        selected_page_contents = []
        selected_page_titles = set() 
        
        logging.info(f"Top candidate pages after combined ranking:")
        for title, score in sorted_candidates[:max_pages + 5]: 
             logging.info(f"  - '{title}' (Combined Score: {score:.4f})")

        # Fetch full content for the top 'max_pages' unique candidates
        for title, score in sorted_candidates:
            if len(selected_page_contents) >= max_pages:
                break
            if title not in selected_page_titles:
                logging.info(f"Fetching content for: '{title}' (Score: {score:.4f})")
                content = self._fetch_page_content_requests(title) 
                time.sleep(0.2) 
                    
                if content and content.strip(): 
                    selected_page_contents.append(content)
                    selected_page_titles.add(title)
                else:
                    logging.warning(f"No content fetched or content was empty for '{title}'.")
        
        logging.info(f"Selected {len(selected_page_contents)} pages for the question.")
        return selected_page_contents
