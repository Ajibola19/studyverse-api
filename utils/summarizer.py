import os
import random
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

def get_groq_keys():
    """Retrieve GROQ_API_KEY_1, GROQ_API_KEY_2, and GROQ_API_KEY_3 from environment."""
    keys = []
    for i in range(1, 4):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)
    return keys

def summarize_text(text):
    """
    Summarize text using key rotation across Groq keys to prevent rate limits during high traffic.
    """
    api_keys = get_groq_keys()
    if not api_keys:
        raise Exception("No Groq API keys found. Please check GROQ_API_KEY_1, GROQ_API_KEY_2, or GROQ_API_KEY_3 in your .env file.")

    # Truncate raw input slightly higher (15,000 chars) so enough context remains to generate 50 questions
    if len(text) > 15000:
        text = text[:15000] + "..."
        print("ℹ️ Truncated text to 15,000 characters to manage context size.")

    # Split into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=3000,
        chunk_overlap=150,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)

    # Fast path for single chunk
    if len(chunks) == 1:
        print("ℹ️ Single chunk document, passing directly for generation.")
        return chunks[0][:6000]

    # Convert chunks to LangChain Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Shuffle key order per request to balance traffic load across active users
    random.shuffle(api_keys)
    last_exception = None

    # Execute chain with fallback key execution
    for idx, key in enumerate(api_keys):
        try:
            print(f"🔄 Executing summarization with Groq Key attempt #{idx + 1}...")

            llm = ChatGroq(
                api_key=key,
                model_name="openai/gpt-oss-20b",
                temperature=0.3
            )

            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce"
            )

            result = chain.invoke(docs)
            summary = result.get("output_text", result) if isinstance(result, dict) else str(result)

            if len(summary) > 8000:
                summary = summary[:8000]
                print("ℹ️ Summary truncated to 8,000 characters.")

            return summary

        except Exception as e:
            print(f"⚠️ Summarizer attempt #{idx + 1} failed: {e}")
            last_exception = e
            continue

    print(f"⚠️ All summarization keys failed ({last_exception}). Falling back to direct chunk text.")
    return chunks[0][:6000]
