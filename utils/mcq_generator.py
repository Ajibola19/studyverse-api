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
    # Use active fast model string
    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant",  # Extremely fast, zero parsing errors
        temperature=0.2
    )
    
    # Force LLM to adhere to structured JSON schema
    structured_llm = llm.with_structured_output(MCQResponse)
    
    prompt = f"""
    Generate {num_questions} multiple-choice questions based on the following text.
    Topic: {topic or 'General'}
    
    Source Text:
    {summary_text}
    """
    
    response = structured_llm.invoke(prompt)
    
    # Convert Pydantic objects directly to Python dicts for Flask
    return [q.model_dump() for q in response.questions]
