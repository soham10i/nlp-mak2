# medqa_project/src/page_selector.py
import wikipedia
import requests
from bs4 import BeautifulSoup
import time
import torch # For tensor operations and type hinting
from sentence_transformers import SentenceTransformer # For type hinting, model might be passed

class PageSelector:
    """
    Selects relevant Wikipedia pages for a given processed question.
    """

    def __init__(self, sentence_model: SentenceTransformer, device: torch.device):
        """
        Initializes the PageSelector.

        Parameters
        ----------
        sentence_model : SentenceTransformer
            The pre-loaded sentence transformer model.
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') the model is on.
        """
        self.sentence_model = sentence_model
        self.device = device
        # The line below caused the AttributeError and has been removed.
        # wikipedia.set_USER_AGENT("MedQAProject/1.0 (contact@example.com; ...) Python-wikipedia/1.4.0")
        # The wikipedia library (1.4.0) does not have a top-level `set_USER_AGENT` function.
        # It uses `requests` internally, which will send a default User-Agent.
        # If specific User-Agent is needed for direct `requests` calls, set it in headers there.
        print("PageSelector initialized. Note: Global Wikipedia User-Agent not set via 'set_USER_AGENT' as the function is unavailable in the library version.")


    def _fetch_page_content_requests(self, title: str) -> str:
        """
        Fetches the full plain text content of a Wikipedia page using requests and BeautifulSoup.
        This method can be more robust for complex pages or if wikipedia.page().content fails.
        """
        headers = {
            'User-Agent': 'MedQAProject/1.0 (your-contact-email@example.com; http://your-project-url.com)'
        }
        # IMPORTANT: Replace with your actual contact email and project URL if you use this method.
        try:
            page = wikipedia.page(title=title, auto_suggest=False, redirect=True)
            url = page.url
            response = requests.get(url, headers=headers, timeout=10) # Added timeout and headers
            response.raise_for_status() # Raise an exception for HTTP errors
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div:
                content_div = soup.find('div', id='bodyContent')

            if content_div:
                for unwanted_tag in content_div.find_all(['table', 'div.infobox', 'div.navbox', 'span.mw-editsection', 'div.thumb', 'div.noprint', 'div.hatnote', 'sup.reference', 'ol.references']):
                    unwanted_tag.decompose()
                
                texts = [p.get_text(separator=' ', strip=True) for p in content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])]
                return "\n".join(filter(None, texts))
            else:
                print(f"Warning: Could not find 'mw-parser-output' for '{title}'. Falling back to wikipedia.page().content.")
                return page.content # This will use the library's default User-Agent

        except wikipedia.exceptions.PageError:
            print(f"Error: Wikipedia page '{title}' not found (PageError).")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Error: '{title}' is a disambiguation page. Options: {e.options[:5]}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page '{title}' with requests: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while fetching content for '{title}': {e}")
        return ""


    def select_pages(self, processed_question_data: dict, max_pages: int = 5, max_candidates_per_query: int = 3) -> list:
        """
        Selects up to max_pages relevant Wikipedia pages for the processed question.

        Parameters
        ----------
        processed_question_data : dict
            The output from QuestionProcessor.process(), containing at least:
            - 'extracted_entities': list of strings
            - 'question_embedding': torch.Tensor
        max_pages : int, optional
            The maximum number of Wikipedia pages to return, by default 5.
        max_candidates_per_query : int, optional
            Max Wikipedia search results to consider per entity query, by default 3.


        Returns
        -------
        list
            A list of strings, where each string is the cleaned text content of a selected Wikipedia page.
        """
        extracted_entities = processed_question_data.get('extracted_entities', [])
        question_embedding = processed_question_data.get('question_embedding')

        if not extracted_entities or question_embedding is None:
            print("Warning: No entities or question embedding provided to PageSelector.")
            return []

        candidate_pages_info = {} 

        search_queries = list(set(extracted_entities))[:10] 

        print(f"\nPageSelector: Using entities for search: {search_queries}")

        for entity_query in search_queries:
            if not entity_query.strip(): 
                continue
            try:
                print(f"Searching Wikipedia for: '{entity_query}'")
                search_results = wikipedia.search(entity_query, results=max_candidates_per_query)
                time.sleep(0.2) # Increased sleep time slightly for politeness

                for page_title in search_results:
                    if page_title not in candidate_pages_info: 
                        try:
                            summary = wikipedia.summary(page_title, sentences=3, auto_suggest=False, redirect=True)
                            time.sleep(0.2) # Increased sleep time
                            if summary:
                                summary_embedding_np = self.sentence_model.encode(summary)
                                summary_embedding = torch.from_numpy(summary_embedding_np).to(self.device)
                                
                                q_emb = question_embedding.unsqueeze(0) if len(question_embedding.shape) == 1 else question_embedding
                                s_emb = summary_embedding.unsqueeze(0) if len(summary_embedding.shape) == 1 else summary_embedding
                                
                                score = torch.nn.functional.cosine_similarity(q_emb, s_emb).item()
                                candidate_pages_info[page_title] = score
                                print(f"  Candidate: '{page_title}', Similarity Score: {score:.4f}")
                        except wikipedia.exceptions.DisambiguationError:
                            print(f"  Skipping disambiguation page: '{page_title}' during summary fetch.")
                        except wikipedia.exceptions.PageError:
                            print(f"  Skipping page (not found for summary): '{page_title}'.")
                        except Exception as e:
                            print(f"  Error processing summary for '{page_title}': {e}")
            except Exception as e:
                print(f"Error during Wikipedia search for '{entity_query}': {e}")
        
        if not candidate_pages_info:
            print("PageSelector: No candidate pages found after searching.")
            return []

        sorted_candidates = sorted(candidate_pages_info.items(), key=lambda item: item[1], reverse=True)
        
        selected_page_contents = []
        selected_page_titles = set() 
        
        print(f"\nPageSelector: Top candidate pages before content fetching:")
        for title, score in sorted_candidates[:max_pages + 5]: 
             print(f"  - '{title}' (Score: {score:.4f})")

        for title, score in sorted_candidates:
            if len(selected_page_contents) >= max_pages:
                break
            if title not in selected_page_titles:
                print(f"Fetching content for: '{title}' (Score: {score:.4f})")
                # Using the more robust _fetch_page_content_requests method by default now
                content = self._fetch_page_content_requests(title) 
                time.sleep(0.2) # Be polite
                    
                if content and content.strip(): 
                    selected_page_contents.append(content)
                    selected_page_titles.add(title)
                else:
                    print(f"  No content fetched or content was empty for '{title}'.")
        
        print(f"\nPageSelector: Selected {len(selected_page_contents)} pages for the question.")
        return selected_page_contents

# Example Usage (for testing this module independently)
if __name__ == '__main__':
    # This import will only work if question_processor.py is in the same directory
    # or src is in PYTHONPATH
    try:
        from src.question_processor import QuestionProcessor 
    except ImportError:
        # Fallback for direct execution if src is not in PYTHONPATH
        # This assumes question_processor.py is in the same directory for standalone testing
        from question_processor import QuestionProcessor
    
    example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Example usage: Running on device: {example_device}")

    try:
        q_processor_for_model = QuestionProcessor(device=example_device)
        page_selector = PageSelector(sentence_model=q_processor_for_model.sentence_model, device=example_device)

        sample_processed_question = {
            "original_question_data": {
                "question": "What are the main symptoms of influenza?"
            },
            "cleaned_question_text": "what are the main symptoms of influenza?",
            "extracted_entities": ["influenza", "symptoms", "main symptoms of influenza"],
            "question_embedding": q_processor_for_model.sentence_model.encode("what are the main symptoms of influenza?", convert_to_tensor=True).to(example_device),
            "question_category": None
        }

        print(f"\n--- Testing PageSelector with sample question ---")
        print(f"Question: {sample_processed_question['original_question_data']['question']}")
        
        selected_contents = page_selector.select_pages(sample_processed_question, max_pages=2)

        if selected_contents:
            print(f"\n--- Content of Selected Pages (first 500 chars each) ---")
            for i, content in enumerate(selected_contents):
                print(f"\nPage {i+1}:")
                print(content[:500] + "...") #.replace('\n', ' '))
        else:
            print("No pages were selected.")

    except Exception as e:
        print(f"An error occurred during PageSelector example usage: {e}")
        import traceback
        traceback.print_exc()
