"""Evaluate the RAG system on a fixed set of questions with known answers."""
import pandas as pd

import config
from rag import answer_question

questions = pd.read_csv("eval/questions.csv", sep=";")
rows = []

for _, q in questions.iterrows():
    out = answer_question(q["question"])
    answer = out["answer"]
    keywords = str(q["expected"]).split("|")
    correct = any(k.lower() in answer.lower() for k in keywords)
    rows.append({
        "question": q["question"],
        "type": q["type"],
        "correct": correct,
        "answer": answer.replace("\n", " ")[:200],
    })
    print(("OK   " if correct else "FAIL ") + q["question"])

result = pd.DataFrame(rows)
result.to_csv(f"eval/results_top{config.TOP_K}.csv", index=False)

print(f"\nTOP_K = {config.TOP_K}")
print(f"Accuracy: {result.correct.sum()}/{len(result)} ({result.correct.mean():.0%})")