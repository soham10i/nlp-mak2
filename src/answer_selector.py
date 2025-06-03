# medqa_project/src/answer_selector.py

class AnswerSelector:
    """
    Selects the best answer from a set of scored options.
    """

    def __init__(self):
        """
        Initializes the AnswerSelector.
        Currently, no specific initialization is needed.
        """
        pass

    def select_answer(self, option_scores: dict) -> str | None:
        """
        Selects the answer option with the highest score.

        Parameters
        ----------
        option_scores : dict
            A dictionary mapping option keys (e.g., 'A', 'B', 'C', 'D')
            to their corresponding scores (float).

        Returns
        -------
        str | None
            The key of the answer option with the highest score.
            Returns None if option_scores is empty or all scores are non-numeric or invalid.
        """
        if not option_scores or not isinstance(option_scores, dict):
            print("Warning: No option scores provided or invalid format. Cannot select answer.")
            return None

        # Filter out any non-numeric scores or handle them if necessary
        valid_scores = {
            key: score for key, score in option_scores.items() 
            if isinstance(score, (int, float))
        }

        if not valid_scores:
            print("Warning: No valid numeric scores found in option_scores. Cannot select answer.")
            return None
            
        # Select the option with the maximum score
        # The `max` function with a `key` argument is suitable here.
        # `valid_scores.get` will be used as the key for comparison.
        try:
            selected_option_key = max(valid_scores, key=valid_scores.get)
            return selected_option_key
        except ValueError: # Handles cases where valid_scores might be empty after filtering
            print("Warning: ValueError during max score selection (e.g. empty valid_scores). Cannot select answer.")
            return None


# Example Usage (for testing this module independently)
if __name__ == '__main__':
    selector = AnswerSelector()

    # Test case 1: Clear winner
    scores1 = {"A": 0.2, "B": 0.8, "C": 0.5, "D": 0.1}
    prediction1 = selector.select_answer(scores1)
    print(f"Scores: {scores1}, Prediction: {prediction1} (Expected: B)")

    # Test case 2: All scores are the same (max returns the first one it encounters or based on dict order in older Python)
    scores2 = {"A": 0.5, "B": 0.5, "C": 0.5}
    prediction2 = selector.select_answer(scores2)
    print(f"Scores: {scores2}, Prediction: {prediction2} (Expected: A or B or C, behavior might vary slightly)")

    # Test case 3: Negative scores
    scores3 = {"A": -0.5, "B": -0.1, "C": -0.8}
    prediction3 = selector.select_answer(scores3)
    print(f"Scores: {scores3}, Prediction: {prediction3} (Expected: B)")
    
    # Test case 4: Empty scores
    scores4 = {}
    prediction4 = selector.select_answer(scores4)
    print(f"Scores: {scores4}, Prediction: {prediction4} (Expected: None)")

    # Test case 5: Scores with non-numeric values (should be handled by filtering)
    scores5 = {"A": 0.7, "B": "invalid", "C": 0.3}
    prediction5 = selector.select_answer(scores5)
    print(f"Scores: {scores5}, Prediction: {prediction5} (Expected: A)")

    # Test case 6: All scores are non-numeric
    scores6 = {"A": "high", "B": "low"}
    prediction6 = selector.select_answer(scores6)
    print(f"Scores: {scores6}, Prediction: {prediction6} (Expected: None)")

    # Test case 7: Mixed valid and None scores (should filter out None)
    scores7 = {"A": 0.1, "B": None, "C": 0.3}
    prediction7 = selector.select_answer(scores7) # This will cause an error if None is not filtered
    # The current implementation filters non-numerics, so it should be fine.
    print(f"Scores: {scores7}, Prediction: {prediction7} (Expected: C)")
