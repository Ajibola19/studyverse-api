import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define strict JSON schema
class Question(BaseModel):
    number: int
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str = Field(description="Must be A, B, C, or D")

class MCQResponse(BaseModel):
    questions: List[Question]

def generate_mcqs(summary_text, num_questions, topic=None):
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name=model_name,
        temperature=0.2
    )
    
    # Enable structured output using json_mode
    structured_llm = llm.with_structured_output(MCQResponse, method="json_mode")
    
    # IMPORTANT: Groq requires the word "json" in the prompt when using json_mode
    prompt = f"""You are an expert exam question creator. Your job is to output structured JSON data.

Generate exactly {num_questions} multiple-choice questions based ONLY on the source text provided. Return the final output strictly as a JSON object matching the requested schema.

Topic: {topic or 'General'}

Source Text:
{summary_text}
"""
    
    try:
        response = structured_llm.invoke(prompt)
        
        # Unpack response
        if response and hasattr(response, 'questions'):
            return [q.model_dump() for q in response.questions]
        elif isinstance(response, dict) and "questions" in response:
            return response["questions"]
        
        return []
    except Exception as e:
        print(f"❌ Error during MCQ generation: {e}")
        raise e
