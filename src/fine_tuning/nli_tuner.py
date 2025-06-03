# medqa_project/src/fine_tuning/nli_tuner.py
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import numpy as np

# Define NLI categories and a mapping
NLI_CATEGORIES = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
nli_label_to_id = {label: i for i, label in enumerate(NLI_CATEGORIES)}
nli_id_to_label = {i: label for i, label in enumerate(NLI_CATEGORIES)}

class NLIDataset(torch.utils.data.Dataset):
    """Custom Dataset class for NLI."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_dummy_nli_data(file_path="dummy_nli_data.jsonl", num_samples=90):
    """Creates a dummy JSONL file for NLI."""
    if os.path.exists(file_path):
        print(f"Dummy NLI data file '{file_path}' already exists. Skipping creation.")
        return

    data = []
    premises = [
        "Insulin therapy is the cornerstone of treatment for all individuals with type 1 diabetes.",
        "Common side effects of metformin include diarrhea and nausea.",
        "The patient presented with a fever and cough.",
        "Aspirin works by inhibiting prostaglandin synthesis.",
        "Lung cancer is primarily caused by smoking.",
        "The sky is blue due to Rayleigh scattering." # Generic example
    ]
    hypotheses_templates = {
        "ENTAILMENT": [
            "Type 1 diabetes requires insulin.",
            "Metformin can cause gastrointestinal issues.",
            "The person showed signs of a respiratory infection.",
            "Aspirin has an anti-inflammatory effect through prostaglandin pathways.",
            "Smoking is a major risk factor for lung cancer.",
            "The blueness of the sky is an atmospheric phenomenon."
        ],
        "NEUTRAL": [
            "Type 1 diabetes is an autoimmune disease.", 
            "Metformin is also used for PCOS.",
            "The patient also had a headache.", 
            "Aspirin was first synthesized in 1897.",
            "Lung cancer treatment involves chemotherapy.",
            "The sky also contains clouds."
        ],
        "CONTRADICTION": [
            "Type 1 diabetes can be managed without insulin.",
            "Metformin has no known side effects.",
            "The patient was completely asymptomatic.",
            "Aspirin promotes prostaglandin synthesis.",
            "Smoking has no link to lung cancer.",
            "The sky is green."
        ]
    }

    for i in range(num_samples):
        premise_idx = i % len(premises)
        label = NLI_CATEGORIES[i % len(NLI_CATEGORIES)]
        hypothesis_idx = i % len(hypotheses_templates[label])
        
        premise = premises[premise_idx]
        hypothesis = hypotheses_templates[label][hypothesis_idx]
        
        data.append({"premise": premise, "hypothesis": hypothesis, "label": label})
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Created dummy NLI data file: {file_path} with {num_samples} samples.")


def load_and_preprocess_nli_data(file_path, tokenizer, label_encoder):
    """Loads NLI data, tokenizes premise-hypothesis pairs, and encodes labels."""
    premises = []
    hypotheses = []
    str_labels = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                premises.append(item['premise'])
                hypotheses.append(item['hypothesis'])
                str_labels.append(item['label'])
    except FileNotFoundError:
        print(f"Error: NLI Data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading or parsing NLI data file {file_path}: {e}")
        return None, None

    if not premises or not str_labels:
        print("No NLI data loaded. Check the data file format and content.")
        return None, None

    try:
        int_labels = label_encoder.transform(str_labels)
    except ValueError as e:
        print(f"Error encoding NLI labels: {e}. Some labels might not be in NLI_CATEGORIES.")
        print(f"Unique labels found in NLI data: {set(str_labels)}")
        print(f"Expected NLI categories: {NLI_CATEGORIES}")
        raise

    encodings = tokenizer(premises, hypotheses, truncation=True, padding=True, max_length=256)
    return encodings, int_labels


def compute_nli_metrics(eval_pred):
    """Computes accuracy for NLI evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

def fine_tune_distilbert_nli(data_file_path: str, model_save_path: str, device: torch.device,
                             num_train_epochs: int = 3, per_device_train_batch_size: int = 8,
                             learning_rate: float = 3e-5):
    """
    Fine-tunes DistilBERT for Natural Language Inference.
    """
    print(f"Starting NLI fine-tuning on device: {device}")
    print(f"Loading NLI data from: {data_file_path}")
    print(f"NLI Model will be saved to: {model_save_path}")

    tokenizer_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        tokenizer_name,
        num_labels=len(NLI_CATEGORIES),
        id2label=nli_id_to_label,
        label2id=nli_label_to_id
    ).to(device)

    label_encoder = LabelEncoder()
    label_encoder.fit(NLI_CATEGORIES) 

    encodings, int_labels = load_and_preprocess_nli_data(data_file_path, tokenizer, label_encoder)
    if encodings is None or int_labels is None:
        print("Failed to load or preprocess NLI data. Aborting fine-tuning.")
        return

    num_samples = len(int_labels)
    if num_samples == 0:
        print("No NLI samples to split. Aborting.")
        return
        
    indices = np.arange(num_samples)
    
    min_samples_for_stratify = len(np.unique(int_labels)) * 2
    
    if num_samples < min_samples_for_stratify and num_samples > 1 :
        print(f"Warning: Not enough samples for NLI stratification. Splitting without stratification.")
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            indices, int_labels, test_size=0.2, random_state=42
        )
    elif num_samples <= 1: 
         print("Warning: Only one or zero NLI samples available. Skipping training.")
         return
    else:
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            indices, int_labels, test_size=0.2, random_state=42, stratify=int_labels
        )

    train_enc_dict = {key: [encodings[key][i] for i in train_indices] for key in encodings.keys()}
    val_enc_dict = {key: [encodings[key][i] for i in val_indices] for key in encodings.keys()}

    train_dataset = NLIDataset(train_enc_dict, train_labels)
    val_dataset = NLIDataset(val_enc_dict, val_labels)

    print(f"NLI Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    if not train_dataset : 
        print("Not enough data to form NLI training dataset. Aborting.")
        return

    # Using modern TrainingArguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_save_path, "nli_checkpoints"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=0.1, 
        weight_decay=0.01,
        logging_dir=os.path.join(model_save_path, "nli_logs"),
        logging_strategy="steps", 
        logging_steps=max(1, int((len(train_dataset) // per_device_train_batch_size) / 10)) if per_device_train_batch_size > 0 and len(train_dataset) > 0 else 10,
        evaluation_strategy="epoch" if len(val_dataset) > 0 else "no",
        save_strategy="epoch", 
        load_best_model_at_end=True if len(val_dataset) > 0 else False,
        metric_for_best_model="accuracy" if len(val_dataset) > 0 else None,
        greater_is_better=True,
        report_to="none", 
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        compute_metrics=compute_nli_metrics if len(val_dataset) > 0 else None,
    )

    print("Starting NLI model training...")
    trainer.train()
    print("NLI model training finished.")

    if len(val_dataset) > 0:
        eval_results = trainer.evaluate()
        print(f"NLI Evaluation results: {eval_results}")

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Fine-tuned NLI model and tokenizer saved to {model_save_path}")

    with open(os.path.join(model_save_path, "nli_label_mappings.json"), "w", encoding='utf-8') as f:
        json.dump({"nli_label_to_id": nli_label_to_id, "nli_id_to_label": nli_id_to_label}, f, indent=4)
    print(f"NLI Label mappings saved to {os.path.join(model_save_path, 'nli_label_mappings.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Natural Language Inference.")
    parser.add_argument("--data_path", type=str, default="../../data/dummy_nli_data.jsonl",
                        help="Path to the JSONL NLI training data file.")
    parser.add_argument("--model_output_path", type=str, default="../../models/nli_classifier",
                        help="Directory to save the fine-tuned NLI model.")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for NLI.")
    
    args = parser.parse_args()
    
    data_dir = os.path.dirname(args.data_path)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created NLI data directory: {data_dir}")

    if not os.path.exists(args.model_output_path):
         os.makedirs(args.model_output_path)
         print(f"Created NLI model output directory: {args.model_output_path}")

    create_dummy_nli_data(args.data_path, num_samples=60) 

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fine_tune_distilbert_nli(
        data_file_path=args.data_path,
        model_save_path=args.model_output_path,
        device=current_device,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr
    )
