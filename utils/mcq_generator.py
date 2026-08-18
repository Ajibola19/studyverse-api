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
    # Retrieve model dynamically or default to active model
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name=model_name,
        temperature=0.2
    )
    
    # FIX: Explicitly set method="json_mode" to avoid "Tool choice required" 400 errors
    structured_llm = llm.with_structured_output(MCQResponse, method="json_mode")
    
    prompt = f"""You are an expert exam question creator.
Generate exactly {num_questions} multiple-choice questions based ONLY on the source text provided.

Topic: {topic or 'General'}

Source Text:
{summary_text}
"""
    
    response = structured_llm.invoke(prompt)
    
    # Handle response and convert to list of dicts for Flask
    if response and hasattr(response, 'questions'):
        return [q.model_dump() for q in response.questions]
    elif isinstance(response, dict) and "questions" in response:
        return response["questions"]
    
    return []
