from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from .env

# Groq retired llama-3.1-8b-instant and llama-3.3-70b-versatile for free and
# developer-tier accounts on 16 Aug 2026. The gpt-oss models are the
# self-serve text models that remain.
LEVELS = {
    # level: (model, reasoning_effort)
    "low": ("openai/gpt-oss-20b", "low"),
    "medium": ("openai/gpt-oss-120b", "low"),
    "high": ("openai/gpt-oss-120b", "medium"),
}


def pick_llm(level: str) -> ChatGroq:
    """
    Picks a Groq-hosted LLM based on the question difficulty.

    Args:
        level (str): "low", "medium", or "high".

    Returns:
        ChatGroq: The LLM instance to be used.
    """
    try:
        model, effort = LEVELS[level.lower()]
    except KeyError:
        raise ValueError(f"Unsupported level: {level}") from None

    return ChatGroq(
        model=model,
        temperature=0,
        max_retries=2,
        reasoning_effort=effort,  # gpt-oss accepts only low / medium / high
    )


if __name__ == "__main__":
    llm_obj = pick_llm("low")
    print(llm_obj.invoke("What is the capital of France?").content)