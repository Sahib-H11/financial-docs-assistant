"""Evaluate the RAG system on questions with known answers (like a model validation).

Usage:
    python evaluate.py            # one run with TOP_K from config.py
    python evaluate.py 4 6 8 10   # compare several TOP_K values ("model specifications")
    python evaluate.py 8 --filter # each question searches only its own report
    python evaluate.py 8 --filter --questions=eval/holdout_questions.csv  # final hold-out test

Metrics per run:
- accuracy            share of questions answered correctly (+ 95% Wilson confidence interval)
- retrieval hit rate  share of questions where the correct page was among the retrieved chunks
- hallucination rate  share of questions with a confident WRONG answer (the dangerous error)
- abstention rate     share of questions where the model said "not found" (the safe error)

BASF was used to choose TOP_K (in-sample). All other documents are out-of-sample.
"""
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config
from ingest import get_vectorstore
from rag import answer_question

TUNED_ON = "BASF_Quartalsmitteilung_03-2026.pdf"  # the document we tuned TOP_K on

# Phrases that mean "I could not find it" (German and English)
ABSTAIN_PHRASES = [
    "cannot find", "can't find", "could not find", "couldn't find", "not find", "not found",
    "does not contain", "do not contain", "no information",
    "keine angabe", "nicht zu finden", "nicht finden", "nicht gefunden", "nicht enthalten",
]


def is_abstention(answer: str) -> bool:
    a = answer.lower()
    if any(p in a for p in ABSTAIN_PHRASES):
        return True
    # German word order: "Diese Information finde ich in den Dokumenten nicht."
    if re.search(r"(finde|kann|liegt).{0,120}\bnicht\b", a):
        return True
    # "Dazu finde ich in den Dokumenten keine Information."
    if re.search(r"\b(finde|kann)\b.{0,80}\bkeine?\b", a):
        return True
    # "Ich kann ... keine Prognose ... finden"
    return bool(re.search(r"\bkeine?\b.{0,120}\b(finden|gefunden|enthalten)\b", a))


def classify(answer: str, expected: str, qtype: str) -> str:
    """Return 'correct', 'abstained' (safe error) or 'wrong' (hallucination)."""
    if answer.startswith("ERROR:"):
        return "error"  # technical failure (e.g. network), not a model mistake
    if qtype == "not_in_document":
        return "correct" if is_abstention(answer) else "wrong"
    if any(k.strip().lower() in answer.lower() for k in str(expected).split("|")):
        return "correct"
    return "abstained" if is_abstention(answer) else "wrong"


def retrieval_hit(sources: list, document: str, pages: str):
    """Was the page with the true answer among the retrieved chunks?"""
    if pd.isna(pages) or str(pages).strip() == "":
        return None  # not applicable (answer is not in any document)
    wanted = {int(float(p)) for p in str(pages).split("|")}  # "3" or "3.0" both work
    return any(s["file"] == document and int(s["page"]) in wanted for s in sources)


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% confidence interval for a proportion (better than +/- for small samples)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def sample_of(document: str) -> str:
    if document == TUNED_ON:
        return "in-sample (tuned)"
    if document == "none":
        return "trap (answer in no document)"
    return "out-of-sample"


def ask_with_retry(question: str, k: int, document: str | None = None, attempts: int = 3) -> dict:
    """Call the RAG system; retry a few times on temporary API errors."""
    for attempt in range(attempts):
        try:
            return answer_question(question, top_k=k, document=document)
        except Exception as e:  # e.g. a temporary network or rate-limit error
            if attempt == attempts - 1:
                return {"answer": f"ERROR: {e}", "sources": []}
            time.sleep(5 * (attempt + 1))


def run(questions: pd.DataFrame, k: int, use_filter: bool = False, workers: int = 6) -> pd.DataFrame:
    """Ask all questions for one TOP_K value, several in parallel to save time."""
    records = questions.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # With --filter, each question searches only its own report (trap questions search all)
        doc_of = lambda q: q["document"] if use_filter and q["document"] != "none" else None
        outputs = list(pool.map(lambda q: ask_with_retry(q["question"], k, doc_of(q)), records))

    rows = []
    for i, (q, out) in enumerate(zip(records, outputs)):
        result = classify(out["answer"], q["expected"], q["type"])
        rows.append({
            "top_k": k,
            "document": q["document"],
            "sample": sample_of(q["document"]),
            "question": q["question"],
            "type": q["type"],
            "result": result,
            "retrieval_hit": retrieval_hit(out["sources"], q["document"], q["pages"]),
            "answer": out["answer"].replace("\n", " ")[:250],
        })
        print(f"  [{i + 1:>2}/{len(records)}] {result.upper():<9} {q['question'][:70]}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    correct = int((df.result == "correct").sum())
    low, high = wilson_ci(correct, n)
    hits = df.retrieval_hit.dropna()
    return {
        "n": n,
        "accuracy": correct / n,
        "ci_low": low,
        "ci_high": high,
        "retrieval_hit_rate": hits.astype(bool).mean() if len(hits) else float("nan"),
        "hallucination_rate": (df.result == "wrong").mean(),
        "abstention_rate": (df.result == "abstained").mean(),
        "errors": int((df.result == "error").sum()),
    }


def main():
    use_filter = "--filter" in sys.argv
    k_values = [int(a) for a in sys.argv[1:] if a.isdigit()] or [config.TOP_K]
    tag = "_filter" if use_filter else ""
    tag += "".join("_" + a[6:] for a in sys.argv[1:] if a.startswith("--tag="))  # e.g. --tag=prompt
    qfile = next((a[12:] for a in sys.argv[1:] if a.startswith("--questions=")), "eval/questions.csv")
    questions = pd.read_csv(qfile, sep=";", dtype={"pages": str})
    if qfile != "eval/questions.csv":
        tag += "_holdout"
    get_vectorstore()  # load the embedding model once, before the parallel calls
    print(f"{len(questions)} questions, {questions.document.nunique()} documents, TOP_K = {k_values}\n")

    details = []
    for k in k_values:
        print(f"TOP_K = {k}")
        details.append(run(questions, k, use_filter))
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(f"eval/results_detail{tag}.csv", index=False)

    summary_rows = []
    for k, g in detail.groupby("top_k"):
        summary_rows.append({"top_k": k, "group": "ALL", **summarize(g)})
        for name, sub in g.groupby("sample"):
            summary_rows.append({"top_k": k, "group": name, **summarize(sub)})
        for name, sub in g[g.document != "none"].groupby("document"):
            summary_rows.append({"top_k": k, "group": name, **summarize(sub)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"eval/summary{tag}.csv", index=False)

    pct = lambda x: "-" if pd.isna(x) else f"{x:.0%}"
    print("\n" + "=" * 96)
    print(f"{'TOP_K':<6}{'group':<40}{'n':>4}{'accuracy':>10}{'95% CI':>13}{'retrieval':>11}{'halluc.':>9}")
    print("-" * 96)
    for _, r in summary.iterrows():
        ci = f"{pct(r.ci_low)}-{pct(r.ci_high)}"
        print(f"{r.top_k:<6}{r.group[:39]:<40}{r.n:>4}{pct(r.accuracy):>10}{ci:>13}"
              f"{pct(r.retrieval_hit_rate):>11}{pct(r.hallucination_rate):>9}")
    print("=" * 96)
    if (detail.result == "error").any():
        print(f"Note: {(detail.result == 'error').sum()} technical errors (API/network), see results_detail.csv")
    print(f"Mode: {'document filter ON' if use_filter else 'no filter (search all documents)'}")
    print(f"Saved: eval/results_detail{tag}.csv (every answer) and eval/summary{tag}.csv (the table above)")


if __name__ == "__main__":
    main()
