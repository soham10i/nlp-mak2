# MedQA Project

This project implements a modular pipeline for MedQA-style question answering.

## Structure

- `main.py`: Orchestrates the pipeline.
- `data/`: Input datasets.
- `src/`: Source code modules.
- `models/`: Fine-tuned models.
- `notebooks/`: Experimentation notebooks.
- `tests/`: Unit/integration tests.

## Setup

### 1. Install dependencies

#### Mac/Linux
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#### Windows
```bat
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
- Place your MedQA dataset in `data/medqa_dataset.jsonl` (or use `medqa_train.jsonl`, `medqa_test.jsonl` as needed).

### 3. Fine-tune Models (Optional but recommended)

#### Question Classifier
```bash
python src/fine_tuning/classifier_tuner.py --data_path data/medqa_train_classifier_labeled.json --model_output_path models/question_classifier
```

#### NLI Model
```bash
python src/fine_tuning/nli_tuner.py --data_path data/medqa_train_nli_labeled.jsonl --model_output_path models/nli_classifier
```

## Usage

Run the main pipeline:
```bash
python main.py --dataset_path data/medqa_test.jsonl --classifier_model_path models/question_classifier --nli_model_path models/nli_classifier
```

- `--dataset_path`: Path to your MedQA test dataset.
- `--classifier_model_path`: Path to your fine-tuned question classifier (optional, but recommended).
- `--nli_model_path`: Path to your fine-tuned NLI model (optional, but recommended).

## How it Works (Sample Example)

1. **Load Data:** The pipeline loads questions and answer options from the MedQA dataset.
2. **Question Classification:** Each question is classified into a category (e.g., DEFINITION, TREATMENT_PROCEDURE, etc.) using the fine-tuned classifier.
3. **Page Selection:** The system extracts key entities from the question and searches Wikipedia for relevant pages.
4. **Evidence Scoring:** Each answer option is scored based on similarity and NLI (Natural Language Inference) with evidence segments from Wikipedia.
5. **Answer Selection:** The option with the highest score is selected as the predicted answer.
6. **Output:** Results are saved to `pipeline_results_nli_integration.jsonl` and accuracy is printed.

### Example Output
```
Processing question 5/5: A 29-year-old woman presents to the clinic after several months of wei...
  EvidenceScorer: Final scores for options: {'A': 0.428, 'B': 0.492, 'C': 0.438, 'D': 0.422}
  Predicted: B, Actual: B, Category: None
  Question 5 processed in 29.26 seconds. Scores: {'A': 0.428, 'B': 0.492, 'C': 0.438, 'D': 0.422}

--- Overall Processing Summary ---
Processed 5 questions.
Total pipeline time: 120.00 seconds.
Average time per question: 24.0000 seconds.

--- Final Accuracy ---
Correct Predictions: 3 / 5
Accuracy: 0.6000

Detailed results saved to pipeline_results_advanced.json
```

## Explanation
- The pipeline processes each question, retrieves evidence, scores options, and predicts the answer.
- The accuracy is calculated as the number of correct predictions divided by the total number of questions.
- You can inspect `pipeline_results_advanced.json` for detailed per-question results.

