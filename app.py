import os
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.pdf_reader import extract_text_from_pdf
from utils.summarizer import summarize_text
from utils.mcq_generator import generate_mcqs
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for PHP backend
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def estimate_tokens(text):
    """Rough estimate: ~4 chars per token"""
    return len(text) // 4

def parse_mcqs_from_text(mcqs_text):
    """
    Fallback parser: Parses raw text output into structured objects
    if the generator returns text instead of dictionaries.
    """
    try:
        questions = []
        current_q = None
        
        lines = mcqs_text.split('\n')
        print(f"\n📝 Fallback Parsing {len(lines)} lines of text...")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match question headers (e.g., Q1., 1., Q50., 50.)
            if re.match(r'^Q?\d+[\.\)]', line, re.IGNORECASE):
                if current_q and 'question' in current_q and 'option_a' in current_q:
                    if all(k in current_q for k in ['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']):
                        questions.append(current_q)
                
                question_text = re.sub(r'^Q?\d+[\.\)]\s*', '', line, flags=re.IGNORECASE).strip()
                current_q = {
                    'number': len(questions) + 1,
                    'question': question_text
                }
            
            # Match options
            elif re.match(r'^[a-dA-D][\.\):]\s*', line):
                if current_q:
                    option_key = line[0].lower()
                    option_text = re.sub(r'^[a-dA-D][\.\):]\s*', '', line).strip()
                    if option_key in ['a', 'b', 'c', 'd']:
                        current_q[f'option_{option_key}'] = option_text
            
            # Match answers
            elif 'answer' in line.lower():
                if current_q:
                    match = re.search(r'[:\s]+([a-dA-D])\b', line)
                    if match:
                        current_q['correct_option'] = match.group(1).upper()
        
        # Save last item
        if current_q and 'question' in current_q and 'option_a' in current_q:
            if all(k in current_q for k in ['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']):
                questions.append(current_q)
        
        return questions
    except Exception as e:
        print(f"❌ Error parsing raw text MCQs: {e}")
        return []

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "API is running", "timestamp": datetime.now().isoformat()})

@app.route("/api/generate-mcqs", methods=["POST"])
def generate_mcqs_api():
    """
    API endpoint to generate MCQs from PDF
    Expected form data:
    - pdf_file: PDF file
    - num_questions: Number of questions to generate
    - topic: Optional topic
    """
    try:
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        num_questions = request.form.get('num_questions', 5)
        topic = request.form.get('topic', '')
        
        try:
            num_questions = int(num_questions)
            if num_questions > 50:
                print(f"⚠️ Requested questions ({num_questions}) exceeded limit. Capping at 50.")
                num_questions = 50
        except ValueError:
            return jsonify({"error": "num_questions must be an integer"}), 400
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        print(f"\n📄 Processing PDF: {filename}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        if not text:
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        print(f"✂️ Extracted {len(text)} characters from PDF")
        
        # Summarize context
        summary = summarize_text(text)
        print(f"\n🔍 SUMMARY ({len(summary)} chars):\n{summary[:300]}...")
        
        # Generate MCQs
        mcqs_output = generate_mcqs(summary, num_questions, topic)
        
        parsed_questions = []
        
        # Case A: mcqs_output is already a list of dicts/objects (Pydantic / Structured Output)
        if isinstance(mcqs_output, list):
            for i, q in enumerate(mcqs_output):
                if isinstance(q, dict):
                    q['number'] = q.get('number', i + 1)
                    parsed_questions.append(q)
        
        # Case B: mcqs_output is a JSON string
        elif isinstance(mcqs_output, str) and mcqs_output.strip().startswith('['):
            try:
                parsed_questions = json.loads(mcqs_output)
            except Exception:
                parsed_questions = parse_mcqs_from_text(mcqs_output)
                
        # Case C: mcqs_output is plain text -> use fallback regex parser
        elif isinstance(mcqs_output, str):
            parsed_questions = parse_mcqs_from_text(mcqs_output)
        
        if not parsed_questions:
            print(f"⚠️ Warning: No questions parsed from output")
            return jsonify({
                "error": "Failed to parse MCQs from API response.",
                "suggestion": "Try requesting fewer questions (e.g., 5 or 10) or uploading a clearer PDF.",
                "raw_sample": str(mcqs_output)[:500]
            }), 500
        
        print(f"✅ Successfully generated {len(parsed_questions)} questions")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "pdf_title": file.filename.replace('.pdf', ''),
            "num_questions": len(parsed_questions),
            "questions": parsed_questions
        }), 200
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}")
        print(traceback.format_exc())
        
        if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
            return jsonify({
                "error": "API rate limit exceeded. Try requesting fewer questions or using a smaller PDF.",
                "suggestion": "Reduce PDF size or number of questions"
            }), 429
        else:
            return jsonify({
                "error": error_msg[:200],
                "type": type(e).__name__
            }), 500

@app.route("/api/test", methods=["GET"])
def test():
    return jsonify({
        "message": "Flask PDF MCQ API is working",
        "endpoints": {
            "health": "/api/health",
            "generate_mcqs": "/api/generate-mcqs (POST with pdf_file, num_questions)"
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
