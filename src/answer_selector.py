class AnswerSelector:
    """
    Selects the best answer from a set of scored options.
    """

    def __init__(self):
        """
        Initializes the AnswerSelector.
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
            return None

        valid_scores = {
            key: score for key, score in option_scores.items()
            if isinstance(score, (int, float))
        }

        if not valid_scores:
            return None

        try:
            selected_option_key = max(valid_scores, key=valid_scores.get)
            return selected_option_key
        except ValueError:
            return None
