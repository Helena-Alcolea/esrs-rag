# esrs-rag

> On-premise **Retrieval-Augmented Generation** over the EU sustainability reporting
> standards (ESRS), with **rigorous, baseline-anchored evaluation** of the retrieval stage.

> 🚧 **Work in progress.** The retrieval pipeline and a hand-verified evaluation set are in
> place; baselines, hybrid retrieval and the generation stage are under construction. Final
> results and figures are not published yet — see [Project status](#project-status).

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
| Retrieval baseline | `BM25` |
| Generation | local LLM via `Ollama` |
| Evaluation | custom metrics (`recall@k`, `MRR`, faithfulness) |
| UI | `Streamlit` |

Built with **explicit code, no orchestration framework** (LangChain/LlamaIndex) — every stage
is spelled out and narrated in the notebooks rather than hidden behind abstractions.

## What's in this repo

A **results showcase**, in the same style as the `spain-weather-forecasting` project — not the
source pipeline:

- **Evaluation notebooks** — corpus EDA, baselines, and the retrieval metrics & ablations, with
  the key pipeline steps shown and explained inline.
- **Precomputed results** (JSON) powering the dashboard.
- A link to the **live Streamlit evaluation dashboard**.

The raw pipeline scripts and the corpus PDFs are not committed.

## Project status

- [x] Corpus extraction — structure-aware, 1,697 records (paragraphs + glossary + appendices)
- [x] Evaluation set — 45 questions, hand-verified against the source paragraph
- [x] Indexing — ChromaDB + e5-base, persistent and local
- [ ] BM25 baseline + retrieval metrics
- [ ] Hybrid retrieval (dense + lexical)
- [ ] Generation stage (Ollama)
- [ ] Evaluation dashboard + final results

## Notes & limitations

- **Legal status of the corpus.** The revised ("Omnibus") ESRS were adopted on 2026-07-03 but
  are not yet in force at the time of writing; Regulation 2023/2772 is the version currently
  reported under, and is the one indexed here. The corpus is the **Annex** (the 12 standards),
  not the articles of the Regulation itself.
- **Out-of-corpus questions.** Questions whose correct answer is "not in the corpus" rest on
  exhaustive search, not on human reading — one cannot read a paragraph that proves an absence.
- **Source anchoring** is to `standard · §paragraph`, which is intrinsic to the document and
  survives re-extraction and changes in chunk size — never to a chunk index.

## License & attribution

The ESRS corpus is official EU legislation (Regulation (EU) 2023/2772), reusable with
attribution. Source: EU Publications Office.
