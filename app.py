import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Financial Document Assistant", layout="wide")
st.title("AI Assistant for Financial Documents")
st.caption("Ask questions about annual reports and tax documents. Every answer cites its source page.")

# --- Sidebar: upload ---
with st.sidebar:
    st.header("1. Upload PDFs")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Index documents") and files:
        for f in files:
            with st.spinner(f"Indexing {f.name} ..."):
                r = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (f.name, f.getvalue(), "application/pdf")},
                    timeout=900,
                )
            if r.ok:
                st.success(f"{f.name}: {r.json()['chunks']} chunks indexed")
            else:
                st.error(f"{f.name}: {r.json().get('detail', r.text)}")

# --- Main area: questions ---
st.header("2. Ask a question")
question = st.text_input("Question (German or English)", placeholder="Wie hoch war die Eigenkapitalquote?")

if st.button("Ask") and question:
    with st.spinner("Searching the documents ..."):
        r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=180)
    if r.ok:
        data = r.json()
        st.markdown(data["answer"])
        st.subheader("Sources")
        for s in data["sources"]:
            with st.expander(f"{s['file']}, page {s['page']}"):
                st.write(s["excerpt"] + " ...")
    else:
        st.error(r.json().get("detail", r.text))