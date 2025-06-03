# medqa_project/main.py
import argparse
import json 
import time 

from src.data_loader import load_medqa_data
from src.utils import get_device
from src.question_processor import QuestionProcessor 
from src.page_selector import PageSelector
from src.evidence_scorer import EvidenceScorer # Already imported
from src.answer_selector import AnswerSelector

def main():
    parser = argparse.ArgumentParser(description="Medical Question Answering System")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the MedQA JSONL dataset file.")
    parser.add_argument("--subset_1000", action="store_true",
                        help="Process only the first 1000 questions.")
    parser.add_argument("--max_pages_per_question", type=int, default=5,
                        help="Max Wikipedia pages per question.")
    parser.add_argument("--scorer_batch_size", type=int, default=32,
                        help="Batch size for EvidenceScorer embeddings.")
    parser.add_argument("--classifier_model_path", type=str, default=None, 
                        help="Path to fine-tuned question classifier model directory.")
    parser.add_argument("--nli_model_path", type=str, default=None, # New argument
                        help="Path to fine-tuned NLI model directory (e.g., models/nli_classifier).")
    parser.add_argument("--top_k_evidence_for_nli", type=int, default=3,
                        help="Number of top similar sentences to pass to NLI model for each option.")

    args = parser.parse_args()

    device = get_device()
    print(f"Loading MedQA data from: {args.dataset_path}")
    all_questions = load_medqa_data(args.dataset_path)

    if not all_questions: print("No questions loaded. Exiting."); return

    if args.subset_1000: questions_to_process = all_questions[:1000]
    else: questions_to_process = all_questions
    print(f"Processing {len(questions_to_process)} questions.")

    if not questions_to_process: print("Question list is empty. Exiting."); return
        
    try:
        q_processor = QuestionProcessor(device=device, classifier_model_path=args.classifier_model_path)
        page_sel = PageSelector(sentence_model=q_processor.sentence_model, device=device) 
        # Pass NLI model path to EvidenceScorer
        evidence_scr = EvidenceScorer(sentence_model=q_processor.sentence_model, 
                                      device=device, 
                                      batch_size=args.scorer_batch_size,
                                      nli_model_path=args.nli_model_path) # Pass NLI path
        ans_sel = AnswerSelector()
    except Exception as e:
        print(f"Failed to initialize processors: {e}"); import traceback; traceback.print_exc(); return 

    pipeline_results_with_scores = [] 
    overall_start_time = time.time()
    total_correct_predictions = 0 

    for i, question_data in enumerate(questions_to_process):
        question_start_time = time.time()
        print(f"\nProcessing question {i+1}/{len(questions_to_process)}: {question_data.get('question', '')[:70]}...")
        
        processed_q_data = q_processor.process(question_data)
        selected_wiki_pages_content = page_sel.select_pages(
            processed_q_data, max_pages=args.max_pages_per_question
        )
        
        option_scores = {} 
        if selected_wiki_pages_content: 
            option_scores = evidence_scr.score_options(
                processed_q_data,
                selected_wiki_pages_content,
                top_k_evidence=args.top_k_evidence_for_nli # Pass k for NLI
            )
        else:
            options_keys = processed_q_data.get('original_question_data', {}).get('options', {}).keys()
            option_scores = {key: 0.0 for key in options_keys}

        predicted_answer_idx = ans_sel.select_answer(option_scores)
        actual_answer_idx = question_data.get('answer_idx')

        if predicted_answer_idx is not None and predicted_answer_idx == actual_answer_idx:
            total_correct_predictions += 1
        
        print(f"  Predicted: {predicted_answer_idx}, Actual: {actual_answer_idx}, Category: {processed_q_data.get('question_category')}")

        pipeline_results_with_scores.append({
            "question_id": i, 
            "original_question": question_data.get('question'),
            "options": question_data.get('options'),
            "actual_answer_idx": actual_answer_idx,
            "predicted_answer_idx": predicted_answer_idx,
            "question_category": processed_q_data.get('question_category'),
            "option_scores": option_scores,
            "pages_found": len(selected_wiki_pages_content)
        })
        question_end_time = time.time()
        print(f"  Question {i+1} processed in {question_end_time - question_start_time:.2f} seconds. Scores: {option_scores}")

    overall_end_time = time.time()
    total_pipeline_time = overall_end_time - overall_start_time
    avg_time_per_question = total_pipeline_time / len(questions_to_process) if questions_to_process else 0
    accuracy = total_correct_predictions / len(questions_to_process) if questions_to_process else 0

    print(f"\n--- Overall Processing Summary ---")
    print(f"Processed {len(pipeline_results_with_scores)} questions.")
    print(f"Total pipeline time: {total_pipeline_time:.2f} seconds.")
    print(f"Average time per question: {avg_time_per_question:.4f} seconds.")
    print(f"\n--- Final Accuracy ---")
    print(f"Correct Predictions: {total_correct_predictions} / {len(questions_to_process)}")
    print(f"Accuracy: {accuracy:.4f}")

    try:
        with open("pipeline_results_nli_integration.jsonl", "w", encoding="utf-8") as f_out:
            for result in pipeline_results_with_scores:
                f_out.write(json.dumps(result) + "\n")
        print("\nDetailed results saved to pipeline_results_nli_integration.jsonl")
    except Exception as e:
        print(f"Error saving results to file: {e}")

if __name__ == "__main__":
    main()
