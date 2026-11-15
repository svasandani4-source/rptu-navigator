"""HiWi finder: resume -> matching research groups (retrieved from official pages) -> draft e-mail."""
from .llm import chat, extract_json
from .retrievers import rrf

PROFILE_PROMPT = """Read the resume. Return JSON only:
{{"summary": "2-sentence profile", "queries": ["5 short search queries, mix of English and German, describing
university research groups / projects / HiWi topics that fit this person"]}}

Resume:
{resume}"""

MATCH_PROMPT = """A student wants a HiWi (student assistant) position. Using ONLY the pages below, pick the 3 best-matching
research groups or projects. For each give: page URL, why it fits (link specific resume skills to specific topics
in the page), and the contact person ONLY if a name appears in the page text (otherwise write "unknown").
Do not invent anything. If fewer than 3 fit, list fewer.

Candidate profile: {summary}

Pages:
{pages}"""

EMAIL_PROMPT = """Write a short e-mail (max 170 words) in {language} from a student to a professor asking about
HiWi / student-assistant openings. Include: subject line, 2 concrete references to the group's work taken from the page
text below, 2-3 relevant skills/projects from the resume, a polite request for a short call or a task, and mention the
attached CV. Do not invent facts. Use [PLACEHOLDERS] for anything unknown (name, availability, hours/week).
Use a salutation with the professor's name only if it appears in the page text.

Resume:
{resume}

Group page ({url}):
{page}"""


def group_by_url(chunks: list[dict], max_pages: int = 5, max_chars: int = 1500) -> list[tuple[str, str]]:
    pages: dict[str, str] = {}
    for c in chunks:
        if c["url"] not in pages and len(pages) >= max_pages:
            continue
        pages[c["url"]] = (pages.get(c["url"], "") + "\n" + c["text"])[:max_chars]
    return list(pages.items())


def find_groups(resume: str, retriever, model: str) -> tuple[str, list[tuple[str, str]]]:
    profile = extract_json(chat(PROFILE_PROMPT.format(resume=resume[:6000]), model)) or {}
    queries = profile.get("queries") or [resume[:300]]
    fused = rrf([retriever.search(q, 15) for q in queries], top=15)
    pages = group_by_url(fused)
    text = "\n\n".join(f"URL: {u}\n{t}" for u, t in pages)
    report = chat(MATCH_PROMPT.format(summary=profile.get("summary", ""), pages=text), model)
    return report, pages


def draft_email(resume: str, url: str, page_text: str, model: str, language: str = "English") -> str:
    return chat(EMAIL_PROMPT.format(language=language, resume=resume[:6000], url=url, page=page_text), model, temperature=0.3)
