import streamlit as st
from pypdf import PdfReader

from navigator.config import CFG
from navigator.hiwi import draft_email, find_groups
from navigator.ingest import load_chunks
from navigator.llm import answer
from navigator.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

st.set_page_config(page_title="RPTU Navigator", layout="wide")
st.title("RPTU Navigator")
st.caption("Answers come only from public RPTU pages and always link to their source. Verify important dates and rules there.")


@st.cache_resource
def build():
    chunks = load_chunks(CFG.corpus_path, CFG.chunk_size, CFG.chunk_overlap)
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, CFG.embed_model)
    return HybridRetriever(bm25, dense)


retriever = build()
k = st.sidebar.slider("Top-k chunks", 3, 10, 5)
lang_choice = st.sidebar.radio("Answer language", ["Auto", "English", "Deutsch"])
tab_qa, tab_hiwi = st.tabs(["Ask RPTU", "HiWi finder"])

with tab_qa:
    q = st.text_input("Question (English or German)", placeholder="How do I re-register for the next semester?")
    if q:
        hits = retriever.search(q, k)
        st.markdown(answer(q, hits, CFG.llm_model, lang_choice))
        st.subheader("Sources")
        for i, h in enumerate(hits, 1):
            with st.expander(f"[{i}] {h['title'] or h['url']}"):
                st.write(h["url"])
                st.write(h["text"])

with tab_hiwi:
    up = st.file_uploader("Resume (PDF or TXT)", type=["pdf", "txt"])
    resume = st.text_area("...or paste your resume text", height=180)
    if up is not None:
        resume = "\n".join(p.extract_text() or "" for p in PdfReader(up).pages) if up.name.endswith(".pdf") else up.read().decode()
    language = st.radio("E-mail language", ["English", "German"], horizontal=True)
    if st.button("Find matching groups") and resume.strip():
        report, pages = find_groups(resume, retriever, CFG.llm_model)
        st.session_state.update(report=report, pages=pages, resume=resume)
    if "report" in st.session_state:
        st.markdown(st.session_state["report"])
        urls = [u for u, _ in st.session_state["pages"]]
        choice = st.selectbox("Draft an e-mail for which page?", urls)
        if st.button("Draft e-mail"):
            text = dict(st.session_state["pages"])[choice]
            st.code(draft_email(st.session_state["resume"], choice, text, CFG.llm_model, language), language="markdown")
            st.warning("Review and personalize before sending. Check names and facts on the group page.")
