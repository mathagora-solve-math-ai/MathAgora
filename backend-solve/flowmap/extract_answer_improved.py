#!/usr/bin/env python3
"""
개선된 답 추출 로직
"""

import re


def extract_answer_improved(content, prob_type):
    """개선된 답 추출

    단답형의 경우 마지막 등호 뒤의 숫자를 찾습니다.
    """
    content = content.strip()

    if prob_type == "5지선다형":
        # 기존 로직과 동일
        match = re.search(r'[①②③④⑤]', content)
        if match:
            char = match.group()
            mapping = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
            return mapping.get(char)

        match = re.search(r'\b([1-5])\b', content)
        if match:
            return match.group(1)

    else:  # 단답형
        # 마지막 등호 뒤의 숫자 찾기 (가장 마지막 등호)
        # 예: "f(8)=8·7·(8+5/8)=56·(69/8)=7·69=483." → 483

        # 마지막 등호 위치 찾기
        last_eq_idx = content.rfind('=')
        if last_eq_idx != -1:
            # 등호 뒤의 내용
            after_eq = content[last_eq_idx+1:].strip()
            # 첫 번째 숫자 찾기
            match = re.search(r'(\d+)', after_eq)
            if match:
                num = int(match.group(1))
                if 0 <= num <= 999:
                    return str(num)

        # Fallback: 마지막 숫자 찾기
        all_numbers = re.findall(r'\b(\d{1,3})\b', content)
        if all_numbers:
            # 마지막 숫자 반환
            num = int(all_numbers[-1])
            if 0 <= num <= 999:
                return str(num)

    return None


if __name__ == "__main__":
    # 테스트
    test_cases = [
        ("f(x)=x(x−1)(x+5/8)이므로\nf(8)=8·7·(8+5/8)=56·(69/8)=7·69=483.", "단답형", "483"),
        ("따라서 답은 ③이다.", "5지선다형", "3"),
        ("최종 답: 32", "단답형", "32"),
    ]

    for content, prob_type, expected in test_cases:
        result = extract_answer_improved(content, prob_type)
        status = "✓" if result == expected else "✗"
        print(f"{status} Expected: {expected}, Got: {result}")
        if result != expected:
            print(f"  Content: {content[:50]}...")
