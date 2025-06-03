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

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place your MedQA dataset in `data/medqa_dataset.jsonl`.

## Usage

Run the main pipeline:
```
python main.py
```
