import os
import random
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_groq_keys():
    """Retrieve GROQ_API_KEY_1, GROQ_API_KEY_2, and GROQ_API_KEY_3 from environment."""
    keys = []
    for i in range(1, 4):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)
            
    return keys

def generate_mcqs(summary_text, num_questions, topic=None):
    # Retrieve available API keys
    api_keys = get_groq_keys()
    if not api_keys:
        raise Exception("No Groq API keys found. Please check GROQ_API_KEY_1, GROQ_API_KEY_2, or GROQ_API_KEY_3 in your .env file.")

    # Get absolute path to prompt file
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

    # Shuffle key order randomly per request to distribute traffic load
    random.shuffle(api_keys)
    last_exception = None

    # Try available keys sequentially if a rate limit or error occurs
    for idx, key in enumerate(api_keys):
        try:
            print(f"🔄 Executing MCQ generation with Groq Key attempt #{idx + 1}...")

            # Set max_tokens to 8000 so up to 50 questions generate without truncation
            llm = ChatGroq(
                api_key=key, 
                model_name="openai/gpt-oss-20b",
                max_tokens=8000,
                temperature=0.4
            )

            chain = prompt | llm | StrOutputParser()

            result = chain.invoke({
                "context": summary_text,
                "num_questions": num_questions,
                "topic": topic or ""
            })

            return result

        except Exception as e:
            print(f"⚠️ Groq Key attempt #{idx + 1} failed or hit rate limits: {e}")
            last_exception = e
            continue

    # Failover fallback error if all 3 keys fail
    raise Exception(f"All 3 Groq API keys failed or hit rate limits. Last error: {last_exception}")
