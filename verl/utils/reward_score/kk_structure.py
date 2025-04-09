import re
import json
from typing import Dict, Tuple, Optional
def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_json: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """   
    # Split response to isolate assistant output
    if "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
        processed_str = processed_str.split("<｜end▁of▁sentence｜>", 1)[0]
        match = re.search(r'\{.*\}', processed_str, re.DOTALL)
        if match:
            processed_str = match.group(0)
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
        processed_str = processed_str.split("<|im_end|>", 1)[0]
        # get content within the outer {}
        match = re.search(r'\{.*\}', processed_str, re.DOTALL)
        if match:
            processed_str = match.group(0)
        
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str
    try:
        solution_json = json.loads(processed_str)
    except json.JSONDecodeError:
        print("[Error] Failed to parse JSON")
        return None, solution_str
    if not "answer" in solution_json:
        print("[Error] No valid answer tags found")
        return None, solution_json
        
    return solution_json, processed_str

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_json: Dict[str, str], expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    item_count = len(answer_json)

    print(f"  Number of predicted roles: {item_count}")
    if item_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {item_count} != {len(expected_names)}")
        return None

    for name in expected_names:        
        if name in answer_json.keys():
            role = answer_json[name]
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict

def validate_response_structure(solution_json: Dict[str, str]) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    if solution_json is None:
        return False
    if "think" in solution_json.keys() and "answer" in solution_json.keys():
        print("Tag sequence validation passed")
        validation_passed = True
    else:
        print("Missing <think>") if not "think" in solution_json else None
        print("Missing <answer>") if not "answer" in solution_json else None
        validation_passed = False

    return validation_passed

def compute_score(solution_str: str, 
                  ground_truth: Dict[str, str],
                  format_reward: int = 1,
                  answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    # to json
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Parse ground truth data
    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")

    # Extract model answer
    solution_json, processed_str = extract_solution(solution_str)
    

    # Validate response structure
    format_correct = validate_response_structure(solution_json)
    if format_correct:
        print(f"\n[Model Response]\n{processed_str}")
    
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    if format_correct and solution_json["answer"]:
        pred_status = parse_model_answer(solution_json["answer"], expected_names)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            
            # score = sum([1 if pred_status[name] == gt_status[name] else 0 for name in expected_names])
            # answer_score = score
            # print(f"Number of correct predictions: {score}")
            if pred_status == gt_status:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                print("  Content validation: MISMATCH")
        else:
            answer_score = -2
            print( "Fail to parse answer")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score