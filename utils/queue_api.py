import os
import re
from datetime import datetime

from flask import Blueprint, jsonify, request

from utils.job_queue import (
    enqueue_job,
    get_job,
    start_job_worker,
    queue_backend_ready,
)
from utils.mcq_generator import generate_mcqs
from utils.pdf_reader import extract_text_from_pdf
from utils.summarizer import summarize_text


queue_api = Blueprint("queue_api", __name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _parse_mcqs(raw_text):
    questions = []
    current = None

    for raw_line in str(raw_text or "").splitlines():
        line = str(raw_line or "").strip()
        line = re.sub(r"^[\\-•*]+\\s*", "", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")

        if not line:
            continue

        match = re.match(
            r"^(?:Q(?:uestion)?\\s*)?(\\d+)\\s*[\\.\\)\\:\\-]\\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if match:
            if current and all(
                key in current
                for key in (
                    "question",
                    "option_a",
                    "option_b",
                    "option_c",
                    "option_d",
                    "correct_option",
                )
            ):
                questions.append(current)

            current = {
                "number": len(questions) + 1,
                "question": match.group(2).strip(),
            }
            continue

        option = re.match(
            r"^(?:option\\s*)?([a-dA-D])\\s*[\\.\\)\\:\\-]\\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if option and current:
            current[f"option_{option.group(1).lower()}"] = option.group(2).strip()
            continue

        answer = re.match(
            r"^(?:answer|correct answer|correct)\\s*[:\\-]?\\s*([a-dA-D])(?:[\\.)])?\\b",
            line,
            re.IGNORECASE,
        )
        if answer and current:
            current["correct_option"] = answer.group(1).upper()

    if current and all(
        key in current
        for key in (
            "question",
            "option_a",
            "option_b",
            "option_c",
            "option_d",
            "correct_option",
        )
    ):
        questions.append(current)

    for index, question in enumerate(questions, start=1):
        question["number"] = index

    return questions

def _process_job(payload):
    file_path = payload["file_path"]
    original_filename = payload["original_filename"]
    num_questions = int(payload["num_questions"])
    topic = payload.get("topic") or ""

    text = extract_text_from_pdf(file_path)
    if not text:
        raise ValueError("Could not extract text from PDF")

    summary = summarize_text(text)
    if not summary:
        raise RuntimeError("Could not prepare PDF content for question generation")

    raw_output = generate_mcqs(summary, num_questions, topic)
    questions = _parse_mcqs(raw_output)

    if len(questions) < num_questions:
        raise RuntimeError(
            f"AI returned only {len(questions)} complete questions; "
            f"{num_questions} were requested"
        )

    # If a provider generates extra valid questions, keep only the number
    # the user requested instead of failing the whole job.
    questions = questions[:num_questions]

    return {
        "success": True,
        "filename": os.path.basename(file_path),
        "pdf_title": original_filename.rsplit(".", 1)[0],
        "num_questions": num_questions,
        "questions": questions,
        "raw_output": raw_output,
    }


def start_queue_worker():
    start_job_worker(_process_job)


@queue_api.route("/api/generate-mcqs-queued", methods=["POST"])
def generate_mcqs_queued():
    ready, queue_error = queue_backend_ready()
    if not ready:
        return jsonify({
            "error": "Shared queue is not configured yet",
            "details": queue_error,
        }), 503

    if "pdf_file" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    file = request.files["pdf_file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        num_questions = int(request.form.get("num_questions", 5))
    except ValueError:
        return jsonify({"error": "num_questions must be an integer"}), 400

    if num_questions < 1:
        return jsonify({"error": "num_questions must be at least 1"}), 400

    num_questions = min(num_questions, 50)
    topic = request.form.get("topic", "")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f_")
    safe_name = os.path.basename(file.filename)
    filename = timestamp + safe_name
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    job_id = enqueue_job(
        {
            "file_path": file_path,
            "original_filename": safe_name,
            "num_questions": num_questions,
            "topic": topic,
        }
    )
    job = get_job(job_id)

    return jsonify(
        {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "queue_position": job.get("queue_position", 1),
        }
    ), 202


@queue_api.route("/api/jobs/<job_id>", methods=["GET"])
def get_mcq_job(job_id):
    ready, queue_error = queue_backend_ready()
    if not ready:
        return jsonify({
            "error": "Shared queue is not configured yet",
            "details": queue_error,
        }), 503

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job), 200
