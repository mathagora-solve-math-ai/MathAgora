#!/usr/bin/env python3
"""Answer Checker for Flow Maps

Extract answers from Final Answer steps and compare with ground truth
"""

import re
import csv


def extract_answer(final_answer_content, prob_type):
    """Extract answer from Final Answer step content

    Args:
        final_answer_content: Content of Final Answer step
        prob_type: "5지선다형" or "단답형"

    Returns:
        Extracted answer as string, or None if not found
    """
    content = final_answer_content.strip()

    if prob_type == "5지선다형":
        # Look for ① ② ③ ④ ⑤ or 1 2 3 4 5
        # Try circled numbers first
        match = re.search(r'[①②③④⑤]', content)
        if match:
            char = match.group()
            # Convert to number
            mapping = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
            return mapping.get(char)

        # Try plain numbers (1-5)
        match = re.search(r'\b([1-5])\b', content)
        if match:
            return match.group(1)

    else:  # 단답형
        # Look for numbers 0-999
        # Try to find the most likely answer (usually at the start or end)
        match = re.search(r'\b(\d{1,3})\b', content)
        if match:
            num = int(match.group(1))
            if 0 <= num <= 999:
                return str(num)

    return None


def check_answer(flowmap, prob_info):
    """Check if model answers are correct

    Args:
        flowmap: Flow map JSON
        prob_info: Problem metadata with 'prob_type' and 'answer'

    Returns:
        Dict: {model: (extracted_answer, is_correct)}
    """
    if not prob_info or 'answer' not in prob_info or 'prob_type' not in prob_info:
        return {}

    ground_truth = str(prob_info['answer']).strip()
    prob_type = prob_info['prob_type']

    results = {}

    # Find Final Answer steps
    for group in flowmap['groups']:
        for step in group['steps']:
            # Check if this is a Final Answer step
            if 'final' in step['title'].lower() or 'answer' in step['title'].lower():
                model = step['model']
                content = step['content']

                # Extract answer
                extracted = extract_answer(content, prob_type)

                if extracted:
                    is_correct = (extracted == ground_truth)
                    results[model] = (extracted, is_correct)

    return results


def extract_choice_content(prob_id, choice_num):
    """Extract choice content from CSV for 5지선다형

    Args:
        prob_id: Problem ID
        choice_num: Choice number (1-5)

    Returns:
        Choice content string, or None if not found
    """
    csv.field_size_limit(10_000_000)
    csv_path = '../data/2024_math_odd.csv'

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['prob_id'] == prob_id:
                    prob_desc = row['prob_desc']

                    # Parse choices (① ... ② ... ③ ... ④ ... ⑤ ...)
                    choices = ['①', '②', '③', '④', '⑤']
                    choice_marker = choices[int(choice_num) - 1]

                    # Find choice content
                    # Pattern: ① <content> ② or ① <content> end
                    pattern = rf'{choice_marker}\s*([^①②③④⑤]+)'
                    match = re.search(pattern, prob_desc)

                    if match:
                        content = match.group(1).strip()
                        # Truncate if too long
                        if len(content) > 30:
                            content = content[:30] + '...'
                        return content
                    break
    except:
        pass

    return None


def get_answer_display(final_answer_content, ground_truth, prob_type, prob_id=None):
    """Get display string for Final Answer

    Args:
        final_answer_content: Raw content from Final Answer step
        ground_truth: Correct answer
        prob_type: "5지선다형" or "단답형"
        prob_id: Problem ID (for extracting choice content)

    Returns:
        Display string with answer and ground truth
    """
    # Format ground truth based on problem type
    if prob_type == "5지선다형" and prob_id:
        # Try to extract choice content
        choice_content = extract_choice_content(prob_id, ground_truth)
        if choice_content:
            formatted_truth = f"{ground_truth}번 ({choice_content})"
        else:
            formatted_truth = f"{ground_truth}번"
    else:
        formatted_truth = ground_truth

    # Show raw content (truncate if too long)
    content_preview = final_answer_content.strip()
    if len(content_preview) > 40:
        content_preview = content_preview[:40] + "..."

    # Escape $ for matplotlib (prevents LaTeX parsing errors)
    content_preview = content_preview.replace('$', r'\$')
    formatted_truth = formatted_truth.replace('$', r'\$')

    return f"Final Answer\n답안: {content_preview}\n정답: {formatted_truth}"


def get_answer_color(model, answer_results):
    """Get color for Final Answer box

    Args:
        model: Model name
        answer_results: Results from check_answer()

    Returns:
        Color string ('green', 'red', or 'black')
    """
    if model not in answer_results:
        return 'black'

    _, is_correct = answer_results[model]
    return 'green' if is_correct else 'red'
