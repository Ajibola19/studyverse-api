from langchain.text_splitter import CharacterTextSplitter

from utils.ai_gateway import iter_text_candidates


MAX_INPUT_CHARS = 8000
MAX_SUMMARY_CHARS = 4000


def summarize_text(text):
    """
    Keep the existing Studyverse input safety ceiling while allowing the
    summarization call itself to fail over across AI providers.
    """
    text = str(text or "")

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + "..."
        print(
            f"ℹ️ Truncated text to {MAX_INPUT_CHARS} characters "
            "as a backend safety limit"
        )

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        return ""

    if len(chunks) == 1:
        print("ℹ️ Single chunk, using text as-is")
        return chunks[0][:3000]

    joined = "\n\n".join(
        f"[SECTION {index}]\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    )

    prompt = f"""
You are preparing source material for difficult university-level MCQs.

Condense the material below into a dense study summary that preserves:
- definitions only when important
- relationships between concepts
- causes and effects
- processes and sequences
- comparisons and distinctions
- exceptions and limitations
- formulas, rules, examples, and technical details that could support exam questions

Do not invent information.
Do not add commentary.
Keep the result below {MAX_SUMMARY_CHARS} characters.

SOURCE MATERIAL:
{joined}
""".strip()

    for candidate in iter_text_candidates(prompt):
        summary = str(candidate["text"] or "").strip()
        if summary:
            print(f"✅ Summary generated with {candidate['model']}")
            return summary[:MAX_SUMMARY_CHARS]

    print("⚠️ All summarization providers unavailable; using source fallback")
    return text[:3000]
