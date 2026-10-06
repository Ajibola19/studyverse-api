import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

from utils.ai_gateway import iter_text_candidates
from utils.mcq_quality import evaluate_mcq_quality, count_complete_mcqs

load_dotenv()


def generate_mcqs(summary_text, num_questions, topic=None):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(current_dir, "prompts", "mcq_prompt.txt")

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")

    with open(prompt_path, encoding="utf-8") as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["context", "num_questions", "topic"],
        template=template,
    )

    rendered_prompt = prompt.format(
        context=summary_text,
        num_questions=num_questions,
        topic=topic or "",
    )

    best_candidate = None
    best_score = -1
    failures = []

    for candidate in iter_text_candidates(rendered_prompt):
        raw_text = candidate["text"]
        model = candidate["model"]

        passed, score, reasons = evaluate_mcq_quality(
            raw_text,
            expected_questions=num_questions,
        )

        print(
            f"🧪 MCQ quality from {model}: score={score}, "
            f"passed={'yes' if passed else 'no'}"
        )

        if passed:
            print(f"✅ Accepted MCQs from {model}")
            return raw_text

        failures.append(
            f"{model}: score={score}; "
            + "; ".join(reasons[:4])
        )

        complete_count = count_complete_mcqs(raw_text)
        if complete_count >= num_questions and score > best_score:
            best_score = score
            best_candidate = raw_text

        print(
            f"⚠️ Rejected {model} output for quality; "
            "trying the next available AI."
        )

    if (
        best_candidate
        and best_score >= 50
        and count_complete_mcqs(best_candidate) >= num_questions
    ):
        print(
            "⚠️ No provider reached the strict MCQ threshold. "
            f"Using best fully-parseable candidate with score {best_score}."
        )
        return best_candidate

    detail = " | ".join(failures[-5:]) if failures else "No AI provider returned usable text."
    raise RuntimeError(
        "All available AI providers failed or produced low-quality MCQs. "
        + detail
    )
