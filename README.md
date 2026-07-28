# esrs-rag

> On-premise **Retrieval-Augmented Generation** over the EU sustainability reporting
> standards (ESRS), with **rigorous, baseline-anchored evaluation** of the retrieval stage.

> 🚧 **Work in progress — not yet finished.** **Retrieval and generation are both evaluated** —
> retrieval against a hand-built BM25 baseline and a hybrid (skill scores, weight sweep); generation
> for answer rate, abstention and hallucination, with **every answer read by a human**. The
> configuration sweeps (`k`, chunking) are done and reported. What remains: the interactive
> dashboard — see [Project status](#project-status).

## What this is

A question-answering assistant over the **12 ESRS standards** (Commission Delegated
Regulation (EU) 2023/2772). You ask a question in plain language; the system retrieves the
relevant paragraphs from the corpus and generates an answer **grounded in those sources and
citing them**.

Why RAG and not a fine-tuned model? Because the regulation *changes* — the ESRS were revised
in July 2026, while this project was being built. A RAG system updates by **re-indexing one
document**; a fine-tuned model would need full retraining and still could not cite its source.

## Why it's different: the evaluation

Most portfolio RAGs are demos with no metrics. The point of this project is to **measure the
system honestly**, the same way a forecasting model is measured against baselines:

- **Retrieval** — `recall@k` and `MRR`, reported as a **skill score against a BM25 baseline**
  (and a trivial random baseline), never as a bare absolute number.
- **Generation** — faithfulness, answer rate, and **abstention** on questions whose answer is
  not in the corpus (a system that says "I don't know" to everything scores perfectly on
  faithfulness and is useless — so answer rate is measured alongside it).
- **A hand-verified evaluation set** of 45 questions, each checked by a human against the exact
  source paragraph, with authorship and verifier recorded per row. No LLM grades its own exam.

Questions are written in **user language**, not the wording of the standard, so that semantic
search has something real to prove over keyword search.

### Language

The primary configuration is **Spanish questions against the Spanish corpus** — the real target
user asks in Spanish. English is kept as a secondary configuration and as an ablation. Retrieval
crosses languages only through the **multilingual embedding model**; there is no separate
translation step, and the generation layer runs *after* retrieval — so it cannot recover a
paragraph that retrieval never found. *Which* corpus best serves a Spanish user is therefore
treated as an empirical question and measured, not assumed.

## Results so far — retrieval

Measured on the 37 evaluation questions that anchor to a specific paragraph, **Spanish questions
against the Spanish corpus** (the primary configuration). A snapshot that may still shift with the
pending ablations. Anchoring is language-invariant, so the comparison across corpora is
apples-to-apples; because chunking is one paragraph per record, the corpora have near-identical size
(1,700 vs 1,697), neutralising the usual length confounder.

- **Dense retrieval (e5-base):** `recall@5 = 0.57`, `recall@10 = 0.65`, `MRR = 0.32`. A modest
  starting point *by design* — a corpus that scored 0.97 out of the box would leave nothing to
  demonstrate.
- **Semantic beats lexical, same language:** BM25 reaches `recall@5 = 0.38`; dense beats it with a
  **skill score of +0.30 over BM25** (and +0.56 over a random baseline).
- **Lexical retrieval collapses across languages:** the same BM25 drops to `recall@5 = 0.05` when
  Spanish questions are run against the *English* corpus — barely above random, because almost no
  content words match. This is the strongest argument for multilingual semantic retrieval, and it
  is measured rather than asserted.
- **Hybrid (dense + BM25 via Reciprocal Rank Fusion), reported honestly:** naive equal-weight
  fusion *hurts* recall — it regresses toward the weaker retriever. A weight sweep recovers a real
  gain in ranking quality (`MRR 0.32 → 0.38`) but **not** in top-5 recall, where the deltas are
  within noise (~1 question out of 37). Reported as-is, not cherry-picked.
- **Contamination bias, quantified:** questions written while reading the source paragraph score
  higher (`recall@10 = 0.86`) than the hand-written user-language questions (`0.60`). That gap is
  the *measured* cost of the bias, reported next to the headline numbers instead of hidden.

## Results so far — generation

Measured on the 45 questions with a local LLM (`llama3.2:3b`, `temperature = 0`), pipeline frozen
(the prompt was tuned only on a throwaway dev set, never on the eval set). **Every answer was read
by a human** against the source paragraph.

- **The faithfulness ↔ answer-rate trade-off, measured.** A system that says *"I don't know"* to
  everything scores a perfect faithfulness of 1.0 and is useless — that was literally the earlier
  demo (**0% answer rate**). The honest measurement tracks **both** ends at once:
  - **Answer rate: 81%** (25/31) on questions that have an answer (the demo was 0%); over-refusal
    (abstaining when it shouldn't) is 19%.
  - **Abstention: 67%** (8/12) on out-of-corpus questions → **hallucination: 33%** (4/12), i.e.
    answering when it should stay silent. This is the next target.
  - **Numeric accuracy: 4/4** on the questions whose answer is a specific figure — gradable by
    string match, with no LLM judge involved.
- **Retrieval and generation are measured separately — and this is why.** One out-of-corpus answer
  was correct *from the model's pre-training* while retrieval had missed the defining paragraph: a
  right answer masking a retrieval failure. Looking only at the final answer would hide it.
- **The abstention detector is a heuristic, hand-checked 12/12.** A metric you don't spot-check is a
  metric you don't trust.
- **How much context to give the model was swept, not guessed.** `k` (paragraphs pasted into the
  prompt) shows an **inverted U**: too few and the answer isn't there; at `k = 20` numeric accuracy
  collapses from 4/4 to 1/4 because the model reads the *neighbouring* paragraph. `k = 8` is frozen
  at the lower edge of the plateau.
- **A measured noise floor for generation (±1–2 questions).** `temperature = 0` is deterministic
  within a run, but re-running an unchanged configuration in a new session can reword an answer —
  enough to flip a phrase-matched check on unchanged content. Differences smaller than that are not
  claimed as effects. Found by re-running a configuration that had not changed.
- **A chunking change that improved retrieval was reverted.** Splitting the ~1% of paragraphs that
  exceed the embedding window raised `recall@5` (0.568 → 0.595, +1 question) — but cost **2 questions
  of answer rate**, because the model then receives a *fragment* and the answer lives in the other
  piece. Net negative end-to-end, so it was rolled back. Measuring retrieval alone would have shipped it.

## Runs 100% on-premise

No external APIs. No credentials anywhere. **No document leaves the machine.** Embeddings run
locally on CPU, the vector store is an on-disk library (no server, telemetry disabled), and
generation runs on a local LLM. For ESG, legal or financial corpora this is a compliance/GDPR
requirement, not a nice-to-have.

## Stack

| Stage | Technology |
|---|---|
| PDF/HTML extraction | `BeautifulSoup` (structure-aware: 1 paragraph = 1 record) |
| Chunking | 1 chunk = 1 numbered paragraph / glossary entry |
| Embeddings | `intfloat/multilingual-e5-base` (512 tokens, 768 dims), local CPU |
| Vector store | `ChromaDB` (persistent, local, telemetry off) |
| Retrieval baseline | `BM25` (hand-implemented, Okapi — no library) |
| Hybrid retrieval | dense + BM25 via Reciprocal Rank Fusion (weight-swept) |
| Generation | local LLM via `Ollama` |
| Evaluation | custom metrics (`recall@k`, `MRR`, faithfulness) |
| UI | `Streamlit` |

Built with **explicit code, no orchestration framework** (LangChain/LlamaIndex) — every stage
is spelled out rather than hidden behind abstractions.

## What's in this repo

A **results showcase**, not the source pipeline:

- **[`notebooks/1-retrieval-evaluation.ipynb`](notebooks/1-retrieval-evaluation.ipynb)** — corpus
  EDA, baselines, retrieval metrics and ablations, presented as **results** (narrative, tables and
  charts). The retrieval implementation itself is kept out; the notebook only reads precomputed
  results and plots them.
- **[`notebooks/2-generation-evaluation.ipynb`](notebooks/2-generation-evaluation.ipynb)** —
  generation quality: the faithfulness ↔ answer-rate trade-off, abstention vs. hallucination, and
  the failure modes surfaced by hand-reading every answer.
- A **Streamlit evaluation dashboard** *(planned — see Project status)*.

The raw pipeline scripts and the corpus PDFs are not committed.

## Project status

- [x] Corpus extraction — structure-aware, 1,697 records (paragraphs + glossary + appendices)
- [x] Evaluation set — 45 questions, hand-verified against the source paragraph
- [x] Indexing — ChromaDB + e5-base, persistent and local (Spanish + English)
- [x] BM25 baseline + retrieval metrics (skill scores vs BM25 and random)
- [x] Hybrid retrieval (dense + lexical) — RRF with a fusion-weight sweep
- [x] Generation stage (Ollama) — measured on 45 questions, **every answer human-verified**
- [x] Generation notebook — the faithfulness ↔ answer-rate trade-off, abstention vs. hallucination
- [x] Configuration sweeps — `k` (inverted U, frozen at 8) and chunking (split variant, reverted)
- [ ] Interactive evaluation dashboard — live semantic search + precomputed answer gallery + metrics

## Notes & limitations

- **Legal status of the corpus.** The revised ("Omnibus") ESRS were adopted on 2026-07-03 but
  are not yet in force at the time of writing; Regulation 2023/2772 is the version currently
  reported under, and is the one indexed here. The corpus is the **Annex** (the 12 standards),
  not the articles of the Regulation itself.
- **Out-of-corpus questions.** Questions whose correct answer is "not in the corpus" rest on
  exhaustive search, not on human reading — one cannot read a paragraph that proves an absence.
- **Source anchoring** is to `standard · §paragraph`, which is intrinsic to the document and
  survives re-extraction and changes in chunk size — never to a chunk index.
- **Small sample, declared noise floors.** Retrieval metrics run on 37 anchored questions: recall
  deltas of ~1 question are treated as noise (improvements below ~0.03 are not trusted). Generation
  runs on 45 questions and carries a **measured ±1–2 question run-to-run wobble**; differences below
  that are not claimed as effects.
- **The configuration was swept on the evaluation set, and that is declared.** `k` and chunking were
  compared over the same 45 questions used for reporting. Both decisions were taken on the *shape and
  mechanism* of the result — the inverted U, the fragment effect — rather than on which value scored
  highest, precisely because the score differences sit inside the noise floors above.

## License & attribution

The ESRS corpus is official EU legislation (Regulation (EU) 2023/2772), reusable with
attribution. Source: EU Publications Office.
