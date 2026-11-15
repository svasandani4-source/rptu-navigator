import json
import os
import re

from dotenv import load_dotenv

load_dotenv()
_client = None


def chat(prompt: str, model: str, system: str = "You are a helpful assistant.", temperature: float = 0.0) -> str:
    import time

    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    from groq import RateLimitError

    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    for attempt in range(4):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=temperature,
                **extra,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            time.sleep(8 * (attempt + 1))
    raise RuntimeError("Groq rate limit hit. Wait a minute and retry, or lower Top-k in the sidebar.")


def extract_json(raw: str):
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None


GERMAN = {"der", "die", "das", "und", "ich", "wie", "was", "ist", "für", "mit", "nicht", "gibt", "welche",
          "kann", "muss", "wo", "wann", "zum", "zur", "bei", "ein", "eine", "mein", "meine", "mir", "ob", "wer"}
ENGLISH = {"the", "is", "what", "how", "do", "my", "tell", "me", "i", "can", "to", "of", "a", "are", "where",
           "when", "which", "for", "with", "and", "in", "on", "does", "have", "need"}


def detect_language(text: str) -> str:
    tokens = re.findall(r"\w+", text.lower())
    de = sum(t in GERMAN for t in tokens) + (2 if re.search(r"[äöüß]", text.lower()) else 0)
    en = sum(t in ENGLISH for t in tokens)
    return "German" if de > en else "English"


ANSWER_PROMPT = """You are an assistant for new students at RPTU Kaiserslautern-Landau.
Answer using ONLY the numbered sources below. The sources may be in German; translate their content as needed.
Rules:
- Write the answer in {language}.
- End every factual sentence with a citation like [1] or [2][3].
- Keep official German terms (e.g. Studierendensekretariat, Prüfungsamt) and, when answering in English,
  add a short English translation in parentheses.
- Never invent dates, deadlines, names or e-mail addresses.
- If the sources do not contain the answer, say so and suggest the relevant office.
- If the question is too vague (e.g. no study program given), say what is missing and give what the sources do say.

Sources:
{context}

Question: {question}
Answer in {language}:"""


def answer(question: str, chunks: list[dict], model: str, language: str = "Auto") -> str:
    if language == "Auto":
        lang = detect_language(question)
    else:
        lang = "German" if language == "Deutsch" else "English"
    context = "\n\n".join(f"[{i}] ({c['url']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    return chat(ANSWER_PROMPT.format(language=lang, context=context, question=question), model)
