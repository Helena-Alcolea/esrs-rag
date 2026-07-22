# ESRS RAG

> On-premise **Retrieval-Augmented Generation** over the EU sustainability reporting
> standards (ESRS), with **rigorous, baseline-anchored evaluation** of the retrieval stage.

> 🚧 **Work in progress.** Retrieval is fully evaluated — dense, a hand-built BM25 baseline, and a
> hybrid, with skill scores and a weight sweep. The generation stage and the evaluation dashboard
> are under construction. Numbers below are a preliminary snapshot — see [Project status](#project-status).

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

## Results so far (retrieval, preliminary)

Measured on the 38 evaluation questions that anchor to a specific paragraph, **Spanish questions
against the Spanish corpus** (the primary configuration). A work-in-progress snapshot, not final
figures. Anchoring is language-invariant, so the comparison across corpora is apples-to-apples;
because chunking is one paragraph per record, the corpora have near-identical size (1,700 vs 1,697),
neutralising the usual length confounder.

- **Dense retrieval (e5-base):** `recall@5 = 0.55`, `recall@10 = 0.61`, `MRR = 0.31`. A modest
  starting point *by design* — a corpus that scored 0.97 out of the box would leave nothing to
  demonstrate.
- **Semantic beats lexical, same language:** BM25 reaches `recall@5 = 0.40`; dense beats it with a
  **skill score of +0.26 over BM25** (and +0.55 over a random baseline).
- **Lexical retrieval collapses across languages:** the same BM25 drops to `recall@5 = 0.05` when
  Spanish questions are run against the *English* corpus — barely above random, because almost no
  content words match. This is the strongest argument for multilingual semantic retrieval, and it
  is measured rather than asserted.
- **Hybrid (dense + BM25 via Reciprocal Rank Fusion), reported honestly:** naive equal-weight
  fusion *hurts* recall — it regresses toward the weaker retriever. A weight sweep recovers a real
  gain in ranking quality (`MRR 0.31 → 0.37`) but **not** in top-5 recall, where the deltas are
  within noise (~1 question out of 38). Reported as-is, not cherry-picked.
- **Contamination bias, quantified:** questions written while reading the source paragraph score
  higher (`recall@10 = 0.86`) than the hand-written user-language questions (`0.55`). That gap is
  the *measured* cost of the bias, reported next to the headline numbers instead of hidden.

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
is spelled out and narrated in the notebooks rather than hidden behind abstractions.

## What's in this repo

A **results showcase**, not the source pipeline:

- **Evaluation notebooks** — corpus EDA, baselines, and the retrieval metrics & ablations, with
  the key pipeline steps shown and explained inline.
- A link to the **live Streamlit evaluation dashboard**.

The raw pipeline scripts and the corpus PDFs are not committed.

## Project status

- [x] Corpus extraction — structure-aware, 1,697 records (paragraphs + glossary + appendices)
- [x] Evaluation set — 45 questions, hand-verified against the source paragraph
- [x] Indexing — ChromaDB + e5-base, persistent and local (Spanish + English)
- [x] BM25 baseline + retrieval metrics (skill scores vs BM25 and random)
- [x] Hybrid retrieval (dense + lexical) — RRF with a fusion-weight sweep
- [ ] Generation stage (Ollama)
- [ ] Evaluation dashboard + final results
- [ ] Cross-lingual ablation (English questions) and chunk-size / `k` sweeps

## Notes & limitations

- **Legal status of the corpus.** The revised ("Omnibus") ESRS were adopted on 2026-07-03 but
  are not yet in force at the time of writing; Regulation 2023/2772 is the version currently
  reported under, and is the one indexed here. The corpus is the **Annex** (the 12 standards),
  not the articles of the Regulation itself.
- **Out-of-corpus questions.** Questions whose correct answer is "not in the corpus" rest on
  exhaustive search, not on human reading — one cannot read a paragraph that proves an absence.
- **Source anchoring** is to `standard · §paragraph`, which is intrinsic to the document and
  survives re-extraction and changes in chunk size — never to a chunk index.
- **Numbers are preliminary.** Retrieval metrics run on 38 anchored questions; recall deltas of
  ~1 question are treated as noise (improvements below ~0.03 are not trusted), and figures may shift
  as the evaluation set grows and chunk-size / `k` are swept.

## License & attribution

The ESRS corpus is official EU legislation (Regulation (EU) 2023/2772), reusable with
attribution. Source: EU Publications Office.
