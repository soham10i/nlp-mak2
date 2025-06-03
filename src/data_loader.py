# medqa_project/src/data_loader.py
import json

def load_medqa_data(file_path):
    """
    Loads the MedQA dataset from a JSONL file.

    Each line in the JSONL file is expected to be a JSON object representing a question.
    Example format: {"question": "...", "answer": "...", "options": {"A": "...", ...}, 
                     "answer_idx": "...", "metamap_phrases": [...]}.

    Parameters
    ----------
    file_path : str
        The path to the MedQA JSONL dataset file.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a question.
        Returns an empty list if the file cannot be read or is not found.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line due to JSON decode error: {e} - Line: '{line.strip()}'")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        # Consider raising the error or handling it more gracefully depending on desired behavior
    except Exception as e:
        print(f"Error loading dataset: {e}")
    return data
