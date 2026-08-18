import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY, 
    model_name="llama-3.3-70b-specdec"
    )

def summarize_text(text):
    """
    Summarize text while keeping token count low for Groq API limits.
    Max summary size: ~2000 tokens to stay within rate limits.
    """
    
    # If text is very large, truncate to first 8000 characters
    # This keeps it manageable for the API
    if len(text) > 8000:
        text = text[:8000] + "..."
        print(f"ℹ️  Truncated text to 8000 characters to avoid token limits")
    
    # Split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    # If only 1 chunk, just process it directly
    if len(chunks) == 1:
        print(f"ℹ️  Single chunk, using text as-is")
        return chunks[0][:3000]  # Still limit the output
    
    # Create documents from chunks
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Use map_reduce to summarize, but with strict token limits
    try:
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            max_tokens=1500  # Keep summary size small
        )
        result = chain.invoke(docs)
        summary = result.get("output_text", result) if isinstance(result, dict) else result
        
        # Ensure summary is not too long
        if len(summary) > 4000:
            summary = summary[:4000]
            print(f"ℹ️  Summary truncated to 4000 characters")
        
        return summary
    except Exception as e:
        print(f"⚠️  Summarization error: {e}")
        # Fallback: use first chunk
        return chunks[0][:3000]
