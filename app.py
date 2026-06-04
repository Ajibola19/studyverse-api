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

# Rough token estimation (for Groq rate limit checking)
def estimate_tokens(text):
    """Rough estimate: ~4 chars per token"""
    return len(text) // 4

def parse_mcqs_from_text(mcqs_text):
    """
    Parse MCQs from the raw text output and return as JSON
    Handles formats like:
    Q1. Question text?
    a. Option A
    b. Option B
    c. Option C
    d. Option D
    Answer: b
    """
    try:
        questions = []
        current_q = None
        
        lines = mcqs_text.split('\n')
        print(f"\n📝 Parsing {len(lines)} lines of text...")
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Match question (Q1., Q2., 1., 2., etc.)
            if any(line.startswith(prefix) for prefix in ['Q1.', 'Q2.', 'Q3.', 'Q4.', 'Q5.', 'Q6.', 'Q7.', 'Q8.', 'Q9.', 'Q10.',
                                                           '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.']):
                # Save previous question if exists
                if current_q and 'question' in current_q and 'option_a' in current_q:
                    # Only add if it has all required fields
                    if all(k in current_q for k in ['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']):
                        questions.append(current_q)
                        print(f"  ✅ Saved Q{len(questions)}: {current_q['question'][:50]}...")
                
                # Extract question text (remove Q prefix and dot)
                question_text = line
                if question_text.startswith('Q'):
                    question_text = question_text[2:]  # Remove 'Q1'
                question_text = question_text.lstrip('. ').strip()
                
                current_q = {
                    'number': len(questions) + 1,
                    'question': question_text
                }
                print(f"  📌 Found Q{len(questions) + 1}: {question_text[:60]}...")
            
            # Match options (case insensitive for a-d)
            elif line and line[0].lower() in 'abcd' and len(line) > 1 and line[1] in '.):':
                if current_q:
                    option_text = line
                    # Remove option prefix (a., a), A., A), etc.)
                    option_text = re.sub(r'^[aAbBcCdD][.):\s]*', '', option_text).strip()
                    
                    option_key = line[0].lower()
                    if option_key == 'a':
                        current_q['option_a'] = option_text
                        print(f"    → Option A: {option_text[:50]}...")
                    elif option_key == 'b':
                        current_q['option_b'] = option_text
                        print(f"    → Option B: {option_text[:50]}...")
                    elif option_key == 'c':
                        current_q['option_c'] = option_text
                        print(f"    → Option C: {option_text[:50]}...")
                    elif option_key == 'd':
                        current_q['option_d'] = option_text
                        print(f"    → Option D: {option_text[:50]}...")
            
            # Match answer line
            elif 'answer' in line.lower():
                if current_q:
                    # Extract the letter AFTER the colon (a, b, c, or d)
                    # Look for patterns like "Answer: b" or "Answer: B" or "Correct: a"
                    match = re.search(r'[:\s]+([aAbBcCdD])', line)
                    if match:
                        current_q['correct_option'] = match.group(1).upper()
                        print(f"    → Correct answer: {match.group(1).upper()}")
                    else:
                        # Fallback: find any letter after colon
                        if ':' in line:
                            after_colon = line.split(':')[1].strip()
                            letter_match = re.search(r'[aAbBcCdD]', after_colon)
                            if letter_match:
                                current_q['correct_option'] = letter_match.group().upper()
                                print(f"    → Correct answer: {letter_match.group().upper()}")
        
        # Add last question
        if current_q and 'question' in current_q and 'option_a' in current_q:
            if all(k in current_q for k in ['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']):
                questions.append(current_q)
                print(f"  ✅ Saved Q{len(questions)}: {current_q['question'][:50]}...")
        
        print(f"\n✅ Successfully parsed {len(questions)} complete questions")
        return questions
    
    except Exception as e:
        print(f"❌ Error parsing MCQs: {e}")
        print(f"Raw output sample: {mcqs_text[:500]}")
        import traceback
        traceback.print_exc()
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
    - topic: Optional topic (for context)
    """
    try:
        # Validate request
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        num_questions = request.form.get('num_questions', 5)
        topic = request.form.get('topic', '')
        
        try:
            num_questions = int(num_questions)
        except ValueError:
            return jsonify({"error": "num_questions must be an integer"}), 400
        
        # Save uploaded file
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
        
        # Check if text is too large (rough estimate)
        text_tokens = estimate_tokens(text)
        if text_tokens > 10000:
            print(f"⚠️  PDF content large ({text_tokens} tokens), will truncate during summarization")
        
        # Summarize text
        summary = summarize_text(text)
        print(f"\n🔍 SUMMARY ({len(summary)} chars):\n{summary[:300]}...")
        
        # Check summary tokens before MCQ generation
        summary_tokens = estimate_tokens(summary)
        if summary_tokens > 8000:
            print(f"⚠️  Summary is large ({summary_tokens} tokens), truncating further...")
            summary = summary[:3000]
            print(f"📝 Truncated summary to 3000 chars")
        
        # Generate MCQs
        mcqs_output = generate_mcqs(summary, num_questions, topic)
        print(f"\n🧠 RAW MCQs OUTPUT:\n{mcqs_output[:500]}...")
        
        # Parse MCQs into structured format
        if isinstance(mcqs_output, str):
            parsed_questions = parse_mcqs_from_text(mcqs_output)
        else:
            parsed_questions = mcqs_output if isinstance(mcqs_output, list) else []
        
        if not parsed_questions or len(parsed_questions) == 0:
            print(f"⚠️ Warning: No questions parsed from output")
            print(f"Raw output sample: {mcqs_output[:800]}")
            return jsonify({
                "error": "Failed to parse MCQs from API response. The format may be unexpected.",
                "suggestion": "Try with a different PDF or fewer questions",
                "raw_sample": mcqs_output[:500]
            }), 500
        
        print(f"✅ Successfully generated {len(parsed_questions)} questions")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "pdf_title": file.filename.replace('.pdf', ''),
            "num_questions": len(parsed_questions),
            "questions": parsed_questions,
            "raw_output": mcqs_output
        }), 200
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}")
        print(traceback.format_exc())
        
        # Check for specific Groq API errors
        if "rate_limit_exceeded" in error_msg.lower() or "413" in error_msg:
            return jsonify({
                "error": "API rate limit exceeded. Your PDF may be too large or complex. Try: (1) Smaller PDF, (2) Fewer questions requested, or (3) Upgrade Groq plan",
                "suggestion": "Reduce PDF size or number of questions"
            }), 429
        elif "tokens" in error_msg.lower():
            return jsonify({
                "error": "Token limit exceeded. Your PDF is too large for the current API tier.",
                "suggestion": "Upload a smaller PDF or split into multiple files"
            }), 429
        else:
            return jsonify({
                "error": error_msg[:200],
                "type": type(e).__name__
            }), 500

@app.route("/api/test", methods=["GET"])
def test():
    """Test endpoint"""
    return jsonify({
        "message": "Flask PDF MCQ API is working",
        "endpoints": {
            "health": "/api/health",
            "generate_mcqs": "/api/generate-mcqs (POST with pdf_file, num_questions)"
        }
    })

if __name__ == "__main__":
    # Uses Render's port if available, otherwise defaults to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
