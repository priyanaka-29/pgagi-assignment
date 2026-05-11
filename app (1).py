# PGAGI Assignment — AI Text Analyzer
# Tools: Python, HuggingFace Transformers, NLP
# Author: Podugu Priyanka

from transformers import pipeline

# ── 1. Sentiment Analysis ──────────────────────────────────────────────────
sentiment_pipe = pipeline("sentiment-analysis",
                          model="distilbert-base-uncased-finetuned-sst-2-english")

# ── 2. Zero-Shot Text Classification ──────────────────────────────────────
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# ── 3. Text Summarization ──────────────────────────────────────────────────
summarizer = pipeline("summarization",
                      model="sshleifer/distilbart-cnn-12-6",
                      max_length=80, min_length=20, do_sample=False)


def analyze_text(text: str, candidate_labels: list) -> dict:
    """
    Runs three NLP tasks on the input text:
      1. Sentiment analysis (positive / negative + confidence score)
      2. Zero-shot topic classification
      3. Abstractive summarization
    Returns a structured result dictionary.
    """
    sentiment  = sentiment_pipe(text)[0]
    category   = classifier(text, candidate_labels)
    summary    = summarizer(text)[0]["summary_text"]

    return {
        "input_text" : text,
        "sentiment"  : {"label": sentiment["label"],
                        "score": round(sentiment["score"] * 100, 2)},
        "top_topic"  : {"label": category["labels"][0],
                        "score": round(category["scores"][0] * 100, 2)},
        "summary"    : summary,
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  INPUT   : {result['input_text'][:80]}...")
    print(f"  SENTIMENT : {result['sentiment']['label']}  "
          f"({result['sentiment']['score']}% confidence)")
    print(f"  TOPIC     : {result['top_topic']['label']}  "
          f"({result['top_topic']['score']}% confidence)")
    print(f"  SUMMARY   : {result['summary']}")
    print("=" * 60)


if __name__ == "__main__":
    LABELS = ["technology", "healthcare", "education", "finance", "environment"]

    samples = [
        ("Artificial intelligence is rapidly transforming every industry. "
         "Machine learning models are now being used in medical diagnosis, "
         "autonomous vehicles, and natural language processing systems that "
         "understand and generate human language with remarkable accuracy."),

        ("Climate change poses an unprecedented threat to our planet. "
         "Rising sea levels, extreme weather events, and loss of biodiversity "
         "are already being felt across the globe. Scientists warn that "
         "immediate action is required to limit temperature rise to 1.5 degrees."),

        ("The stock market experienced significant volatility this quarter "
         "as investors reacted to rising interest rates and inflation data. "
         "Technology stocks led the decline, while energy and utility sectors "
         "showed resilience amid global supply chain uncertainties."),
    ]

    print("\n🤖  PGAGI AI Text Analyzer — Podugu Priyanka")
    print("    Tools: Python | HuggingFace Transformers | NLP")

    for text, _ in samples:
        result = analyze_text(text, LABELS)
        print_result(result)

    print("\n✅  Analysis complete.")
