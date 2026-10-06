import re
from difflib import SequenceMatcher


def _normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def _clean_line(line):
    line = str(line or "").strip()
    line = re.sub(r"^[\-•*]+\s*", "", line)
    return line.replace("**", "").replace("__", "").replace("`", "").strip()


def _parse_questions(raw_text):
    questions = []
    current = None

    for raw_line in str(raw_text or "").splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue

        question_match = re.match(
            r"^(?:Q(?:uestion)?\s*)?(\d+)\s*[\.\)\:\-]\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if question_match:
            if current:
                questions.append(current)
            current = {
                "question": _normalize(question_match.group(2)),
                "options": {},
                "answer": None,
            }
            continue

        option_match = re.match(
            r"^(?:option\s*)?([a-dA-D])\s*[\.\)\:\-]\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if option_match and current:
            current["options"][option_match.group(1).upper()] = _normalize(
                option_match.group(2)
            )
            continue

        answer_match = re.match(
            r"^(?:answer|correct answer|correct)\s*[:\-]?\s*([a-dA-D])(?:[\.)])?\b",
            line,
            re.IGNORECASE,
        )
        if answer_match and current:
            current["answer"] = answer_match.group(1).upper()

    if current:
        questions.append(current)

    return questions


def _option_similarity(options):
    pairs = []
    values = list(options.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            pairs.append(
                SequenceMatcher(None, values[i].lower(), values[j].lower()).ratio()
            )
    return max(pairs) if pairs else 0.0


def count_complete_mcqs(raw_text):
    questions = _parse_questions(raw_text)
    return sum(
        1
        for q in questions
        if set(q.get("options", {}).keys()) == {"A", "B", "C", "D"}
        and q.get("answer") in {"A", "B", "C", "D"}
    )


def evaluate_mcq_quality(raw_text, expected_questions):
    questions = _parse_questions(raw_text)
    reasons = []
    score = 100

    if len(questions) != expected_questions:
        reasons.append(
            f"expected {expected_questions} questions but parsed {len(questions)}"
        )
        score -= min(40, abs(expected_questions - len(questions)) * 8)

    longest_correct_count = 0
    analyzable = 0
    repeated_answer_positions = []

    for index, q in enumerate(questions[:expected_questions], start=1):
        options = q.get("options", {})
        answer = q.get("answer")

        if set(options.keys()) != {"A", "B", "C", "D"}:
            reasons.append(f"Q{index}: missing one or more options")
            score -= 18
            continue

        if answer not in options:
            reasons.append(f"Q{index}: invalid or missing answer key")
            score -= 18
            continue

        analyzable += 1
        repeated_answer_positions.append(answer)

        lengths = {key: max(1, len(value.split())) for key, value in options.items()}
        shortest = min(lengths.values())
        longest = max(lengths.values())
        correct_len = lengths[answer]
        sorted_lengths = sorted(lengths.values())
        median_like = (sorted_lengths[1] + sorted_lengths[2]) / 2

        # Length imbalance is a quality warning, not a reason to discard
        # an otherwise complete quiz.
        if longest / shortest > 2.0:
            reasons.append(f"Q{index}: option lengths are too uneven")
            score -= 3

        if correct_len == longest and correct_len > median_like * 1.5:
            longest_correct_count += 1
            reasons.append(f"Q{index}: correct answer is conspicuously longer")
            score -= 4
        elif correct_len == longest:
            longest_correct_count += 1

        normalized_options = [value.lower().strip() for value in options.values()]
        if len(set(normalized_options)) != 4:
            reasons.append(f"Q{index}: duplicate options")
            score -= 15

        if _option_similarity(options) > 0.95:
            reasons.append(f"Q{index}: two options are almost duplicates")
            score -= 6

        banned = ("all of the above", "none of the above")
        if any(any(term in opt.lower() for term in banned) for opt in options.values()):
            reasons.append(f"Q{index}: contains all/none-of-the-above shortcut")
            score -= 4

        if len(q.get("question", "").split()) < 4:
            reasons.append(f"Q{index}: question is too shallow/short")
            score -= 3

    if analyzable >= 4:
        ratio = longest_correct_count / analyzable
        if ratio > 0.70:
            reasons.append(
                "the correct answer is the longest option too often across the set"
            )
            score -= 7

        if len(set(repeated_answer_positions)) == 1:
            reasons.append("every correct answer uses the same option position")
            score -= 10
        elif analyzable >= 8:
            max_position_share = max(
                repeated_answer_positions.count(letter) for letter in "ABCD"
            ) / analyzable
            if max_position_share > 0.65:
                reasons.append("correct-answer positions are too predictable")
                score -= 5

    score = max(0, min(100, score))

    passed = (
        len(questions) >= expected_questions
        and analyzable >= expected_questions
        and score >= 70
    )

    return passed, score, reasons
