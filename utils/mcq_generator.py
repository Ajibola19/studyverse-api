import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_mcqs(summary_text, num_questions, topic=None):
    # Get the absolute path to the prompt file
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(current_dir, "prompts", "mcq_prompt.txt")
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["context", "num_questions", "topic"],
        template=template,
    )

    # Added temperature=0.3 to enforce strict adherence to equal option lengths
    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name="openai/gpt-oss-120b",
        temperature=0.3
    )
    
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "context": summary_text,
        "num_questions": num_questions,
        "topic": topic or "General"
    })
    
    return result
