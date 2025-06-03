# medqa_project/src/fine_tuning/classifier_tuner.py
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import numpy as np

# Define categories and a mapping
CATEGORIES = ["DEFINITION", "TREATMENT_PROCEDURE", "DIAGNOSIS_SYMPTOM", "CAUSE_MECHANISM", "OTHER"]
label_to_id = {label: i for i, label in enumerate(CATEGORIES)}
id_to_label = {i: label for i, label in enumerate(CATEGORIES)}

class MedQADataset(torch.utils.data.Dataset):
    """Custom Dataset class for MedQA question classification."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_dummy_labeled_data(file_path="dummy_q_classification_data.jsonl", num_samples=100):
    """Creates a dummy JSONL file for question classification."""
    if os.path.exists(file_path):
        print(f"Dummy data file '{file_path}' already exists. Skipping creation.")
        return

    data = []
    questions = [
        "What is diabetes mellitus type 2?",
        "How is acute myocardial infarction treated?",
        "What are the common symptoms of influenza?",
        "Why does smoking cause lung cancer?",
        "When was penicillin discovered?",
        "Describe the mechanism of action for aspirin.",
        "Which diagnostic test is used for tuberculosis?",
        "What is the prognosis for stage III colon cancer?",
        "Explain the pathophysiology of asthma.",
        "List common side effects of lisinopril."
    ]
    for i in range(num_samples):
        question = questions[i % len(questions)] + f" (Sample {i})"
        label = CATEGORIES[i % len(CATEGORIES)]
        data.append({"text": question, "label": label})
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Created dummy data file: {file_path} with {num_samples} samples.")


def load_and_preprocess_data(file_path, tokenizer, label_encoder):
    """Loads data from JSONL, tokenizes, and encodes labels."""
    texts = []
    str_labels = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: 
            for line in f:
                item = json.loads(line)
                texts.append(item['text'])
                str_labels.append(item['label'])
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None, None 
    except Exception as e:
        print(f"Error reading or parsing data file {file_path}: {e}")
        return None, None, None 

    if not texts or not str_labels:
        print("No data loaded. Check the data file format and content.")
        return None, None, None 

    try:
        int_labels = label_encoder.transform(str_labels)
    except ValueError as e:
        print(f"Error encoding labels: {e}. Some labels might not be in CATEGORIES.")
        print(f"Unique labels found in data: {set(str_labels)}")
        print(f"Expected categories: {CATEGORIES}")
        raise

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return encodings, int_labels, texts 


def compute_metrics(eval_pred):
    """Computes accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

def fine_tune_distilbert_classifier(data_file_path: str, model_save_path: str, device: torch.device,
                                    num_train_epochs: int = 3, per_device_train_batch_size: int = 8,
                                    learning_rate: float = 5e-5):
    """
    Fine-tunes DistilBERT for question classification.
    """
    print(f"Starting fine-tuning for question classification on device: {device}")
    print(f"Loading data from: {data_file_path}")
    print(f"Model will be saved to: {model_save_path}")

    tokenizer_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        tokenizer_name,
        num_labels=len(CATEGORIES),
        id2label=id_to_label,
        label2id=label_to_id
    ).to(device)

    label_encoder = LabelEncoder()
    label_encoder.fit(CATEGORIES) 

    encodings, int_labels, _ = load_and_preprocess_data(data_file_path, tokenizer, label_encoder)
    if encodings is None or int_labels is None:
        print("Failed to load or preprocess data. Aborting fine-tuning.")
        return

    num_samples = len(int_labels)
    if num_samples == 0:
        print("No samples to split. Aborting.")
        return
        
    indices = np.arange(num_samples)
    
    min_samples_for_stratify = len(np.unique(int_labels)) * 2 
    
    if num_samples < min_samples_for_stratify and num_samples > 1 : 
        print(f"Warning: Not enough samples or classes for stratification (samples: {num_samples}, unique labels: {len(np.unique(int_labels))}). Splitting without stratification.")
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            indices, int_labels, test_size=0.2, random_state=42 
        )
    elif num_samples <= 1: 
         print("Warning: Only one or zero samples available. Cannot split into train/validation. Skipping training.")
         return
    else: 
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            indices, int_labels, test_size=0.2, random_state=42, stratify=int_labels
        )

    train_enc_dict = {key: [encodings[key][i] for i in train_indices] for key in encodings.keys()}
    val_enc_dict = {key: [encodings[key][i] for i in val_indices] for key in encodings.keys()}

    train_dataset = MedQADataset(train_enc_dict, train_labels)
    val_dataset = MedQADataset(val_enc_dict, val_labels)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    if not train_dataset : 
        print("Not enough data to form a training dataset after split. Aborting.")
        return

    # Using modern TrainingArguments (compatible with transformers >=4.4.0)
    training_args = TrainingArguments(
        output_dir=os.path.join(model_save_path, "checkpoints"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=0.1, # Using warmup_ratio instead of calculating warmup_steps manually
        weight_decay=0.01,
        logging_dir=os.path.join(model_save_path, "logs"),
        logging_strategy="steps",
        logging_steps=max(1, int((len(train_dataset) // per_device_train_batch_size) / 10)) if per_device_train_batch_size > 0 else 10,
        evaluation_strategy="epoch" if len(val_dataset) > 0 else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if len(val_dataset) > 0 else False,
        metric_for_best_model="accuracy" if len(val_dataset) > 0 else None,
        greater_is_better=True,
        report_to=["none"],  # Use list for report_to in latest transformers
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        compute_metrics=compute_metrics if len(val_dataset) > 0 else None,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    if len(val_dataset) > 0:
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Fine-tuned model and tokenizer saved to {model_save_path}")

    with open(os.path.join(model_save_path, "label_mappings.json"), "w", encoding='utf-8') as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=4)
    print(f"Label mappings saved to {os.path.join(model_save_path, 'label_mappings.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Question Classification.")
    parser.add_argument("--data_path", type=str, default="../../data/dummy_q_classification_data.jsonl",
                        help="Path to the JSONL training data file.")
    parser.add_argument("--model_output_path", type=str, default="../../models/question_classifier",
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    
    args = parser.parse_args()
    
    data_dir = os.path.dirname(args.data_path)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")

    if not os.path.exists(args.model_output_path):
         os.makedirs(args.model_output_path)
         print(f"Created model output directory: {args.model_output_path}")

    create_dummy_labeled_data(args.data_path, num_samples=50) 

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fine_tune_distilbert_classifier(
        data_file_path=args.data_path,
        model_save_path=args.model_output_path,
        device=current_device,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr
    )
