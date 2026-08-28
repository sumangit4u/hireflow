# HireFlow — AI-Powered Resume Search

HireFlow indexes PDF resumes using hybrid search (BM25 + Pinecone vectors), fuses results with Reciprocal Rank Fusion, applies post-search filters, and adds a qualitative Gemini assessment (strengths/gaps) to the top results.

---

## Setup

HireFlow lives inside the **Module 2 - RAG** workspace and shares its dependencies.
There is no per-project `requirements.txt` — everything is declared in
`../pyproject.toml`, pinned in `../uv.lock`, and managed with
[uv](https://docs.astral.sh/uv/).

### 1. Install dependencies with uv

Run this from the **module root** — the directory holding `pyproject.toml`, one level above `Hireflow/`:

```bash
cd "Module 2 - RAG"
uv sync
```

`uv sync` creates `.venv/` at the module root and installs the exact locked
dependency set. It also downloads CPython 3.11 if you don't already have it: the
project pins `requires-python = ">=3.11,<3.12"` (see `.python-version`).

You never need to activate the virtualenv. Prefix commands with `uv run` and uv
picks the right environment automatically, including from inside `Hireflow/`:

```bash
cd Hireflow
uv run python -c "import langchain; print(langchain.__version__)"
```

If you'd still rather have an activated shell:

```bash
source ../.venv/bin/activate        # Windows: ..\.venv\Scripts\activate
```

To add a dependency, run `uv add <package>` from the module root (or edit
`../pyproject.toml` and re-run `uv sync`) — both refresh `uv.lock`.

### 2. Configure environment variables

Create a `.env` file at the **module root**, next to `pyproject.toml` — not inside `Hireflow/`:

```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_key
```

Everything else has a default in `utils/config.py`; set these only to override:

```
PINECONE_INDEX_NAME=hireflow          # default: hireflow
PINECONE_DIMENSION=384                # default: 384 (matches all-MiniLM-L6-v2)
PINECONE_METRIC=cosine                # default: cosine
GOOGLE_MODEL=gemini-2.5-flash-lite    # default: gemini-2.5-flash-lite
```

### 3. Add resumes

Place your PDF resumes in `data/resumes/`.

### 4. Build the index — the required first step

**Index the resumes before starting anything else.** Nothing else in HireFlow
indexes on your behalf: the Streamlit app and the API both read an index that
already exists, and neither will find candidates until you build one.

From the `Hireflow/` directory:

```bash
uv run python index_resumes.py
```

This is the only expensive operation in the project — it reads every PDF and
makes **one Gemini call per resume** to extract name, skills, location, and
experience, then embeds each resume and upserts it into Pinecone. For 50
resumes expect ~50 LLM calls and a couple of minutes. It runs **once**.

The result is written to `data/hybrid_index/` and reused from then on, so app
startup costs zero API calls. Re-run the same command whenever you add resumes:
it is incremental and only parses PDFs that are new or changed.

```bash
uv run python index_resumes.py --status    # what's indexed; does no work
uv run python index_resumes.py             # incremental — new/changed PDFs only
uv run python index_resumes.py --force     # full rebuild, re-parses everything
```

Use `--force` only when you actually want to pay for a full re-parse — after
changing the parsing prompt or schema, or to recover a corrupted index.

---

## How indexing works

Indexing is deliberately separated from running the app, because the two have
very different costs:

| Step | When it runs | Cost |
|---|---|---|
| **Index** (`index_resumes.py`) | Explicitly, when you run it | PDF read + 1 Gemini call + 1 embedding per resume |
| **Start the app** | Every launch | Reads `data/hybrid_index/`, rebuilds BM25 in memory. **No PDF reads, no LLM calls, no Pinecone writes** |
| **Search** | Per query | 1 embedding for the query + 1 Pinecone query |

What gets persisted in `data/hybrid_index/`:

- `corpus.pkl` — resume texts plus their parsed metadata, everything needed to
  rebuild the BM25 index in memory in milliseconds.
- `manifest.json` — one entry per indexed PDF with its SHA-256. This is how
  incremental runs know what to skip. Hashing content rather than checking
  mtime means copying or re-downloading an unchanged resume costs nothing.

Adding 2 PDFs to a folder of 50 costs 2 Gemini calls, not 52. Deleting a PDF
drops it from the index on the next run. Resumes uploaded through the Streamlit
UI are parsed once, appended to the index, and persisted — so they survive a
restart and are not re-parsed later.

---

## Feature Walkthrough

Follow these steps in order. Each one builds on the previous.
**Run every command from the `Hireflow/` directory** — `index_resumes.py`,
`evaluate_retrieval.py`, and `start_backend.py` resolve their imports relative to it.

### Feature 1 — Run the test suite

```bash
uv run pytest tests/ -v
```

All tests are self-contained: Pinecone and Gemini are mocked, so nothing touches an
external API. Coverage spans `test_filters`, `test_hybrid_indexer`,
`test_candidate_evaluator`, `test_ingestion`, `test_indexing_service` (which pins
down the "startup makes zero LLM calls" guarantee), and `test_retrieval_eval`
(worked examples of MRR / MAP / NDCG).

---

### Feature 2 — Start the FastAPI backend

```bash
uv run python start_backend.py
```

This starts the API at `http://localhost:8000`. Open `http://localhost:8000/docs` to see the Swagger UI.

Equivalent, if you'd rather drive uvicorn yourself:

```bash
uv run uvicorn api.main:app --reload --port 8000
```

The backend loads the persisted index on first use, so it comes up searchable —
you do **not** need to call `/index` after every restart.

**Check system status:**

```bash
curl http://localhost:8000/status
```

```json
{
  "resumes_ready": true,
  "vector_store_ready": true,
  "hybrid_ready": true,
  "pinecone_vector_count": 50,
  "indexed_resumes": 50,
  "last_indexed": "2026-08-28T19:04:11+00:00",
  "unindexed_pdfs": 0
}
```

`unindexed_pdfs` counts PDFs sitting in `data/resumes/` that the index doesn't
cover yet.

---

### Feature 3 — Re-index via the API

The same indexing service as the CLI, exposed over HTTP. Incremental by default:

```bash
curl -X POST http://localhost:8000/index
```

```json
{"indexed": 50, "message": "Index ready: 2 parsed, 48 skipped, 50 total in index"}
```

Force a full re-parse (one Gemini call per resume):

```bash
curl -X POST "http://localhost:8000/index?force=true"
```

---

### Feature 4 — Search candidates via the API

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Senior Accountant with QuickBooks", "top_k": 5}'
```

Response includes three separate scores for each candidate:

```json
{
  "results": [
    {
      "candidate_id": "c_alice_smith",
      "name": "Alice Smith",
      "bm25_score": 0.8721,
      "vector_score": 0.7534,
      "combined_score": 0.0323,
      "skills": ["QuickBooks", "Excel", "Accounting"],
      "location": "New York, USA",
      "experience": 5
    }
  ],
  "total": 5
}
```

- **bm25_score** — normalized keyword match (0-1)
- **vector_score** — cosine similarity from Pinecone (0-1)
- **combined_score** — Reciprocal Rank Fusion across both lists

---

### Feature 5 — Launch the Streamlit web UI

Open a second terminal:

```bash
uv run streamlit run streamlit/app.py
```

**Startup does no indexing.** The app loads `data/hybrid_index/` and rebuilds
BM25 in memory — a local file read, no PDF parsing, no Gemini calls, no Pinecone
writes. Launching is fast and free no matter how many resumes you have.

If you haven't built an index yet, the app says so and shows the command instead
of silently indexing 50 resumes for you. The sidebar shows when the index was
built and warns when `data/resumes/` holds PDFs the index doesn't cover.

---

### Feature 6 — Upload new resumes via the UI

In the Streamlit app:
1. Use the **Add Resumes** panel on the left
2. Upload one or more PDF files
3. Click **Process & Index Resumes**

Each uploaded resume is saved to `data/resumes/`, parsed once with Gemini to
extract name, skills, location, and experience, then added to BM25 + Pinecone
**and written to the persisted index**. It survives a restart, and a later
`index_resumes.py` run will not re-parse it. Re-uploading a resume with the same
filename replaces its entry rather than duplicating it.

---

### Feature 7 — Search with filters

In the **Search Candidates** panel:
1. Enter a **Job Title** (e.g. "Senior Accountant")
2. Enter a **Job Description** (e.g. "Looking for experienced accountant with tax expertise")
3. Enter **Required Skills** (one per line)
4. Optionally set a **Location Filter** and **Min. Experience**
5. Click **Find Candidates**

Post-search filters (`core/filters.py`) are applied after the hybrid search to narrow down by skills, location, and experience. Results show:
- BM25 Score, Vector Score, and Combined (RRF) as separate metrics
- Skills, experience, and location for each candidate
- Resume preview

---

### Feature 8 — AI evaluation

For the top 5 search results, the **CandidateEvaluator** (`core/candidate_evaluator.py`) calls Gemini to produce:
- **Strengths** — what makes this candidate a good match
- **Gaps** — where they fall short
- **Summary** — one-line assessment

These appear inside each candidate's expandable card. If Gemini is unavailable, a
rule-based fallback reports skill overlap instead.

**There is no AI fit score.** Candidates are ranked by the three retrieval scores
only — BM25, vector, and their RRF combination. An LLM-derived 0-100 number used to
sit alongside them, computed on a completely different basis, which made it unclear
which number actually drove the ordering. The qualitative findings stay because they
explain a match without pretending to measure it.

---

### Feature 9 — Offline retrieval evaluation

Sidebar > **Retrieval Evaluation**, or from the terminal:

```bash
uv run python evaluate_retrieval.py            # summary
uv run python evaluate_retrieval.py --detail   # every hit with its three scores
uv run python evaluate_retrieval.py --top-k 5  # score the top 5 instead of 10
uv run python evaluate_retrieval.py --list-skills
```

*Offline* means the correct answers are known in advance, so no human judge and
no LLM are involved — this measures **retrieval**, not answer quality. It needs
only the existing index and makes no Gemini calls.

Each test case in `data/eval/test_cases.json` pairs a query with the skills a
candidate must have to count as relevant:

```json
{"query": "tax preparation specialist", "relevant_skills": ["tax preparation"]}
```

Ground truth is read off the indexed metadata: every candidate whose parsed
skills contain all the listed skills. Worth saying plainly — retrieval searches
the resume *text* while relevance is judged on the *skills Gemini extracted*,
so the metrics ask whether text search surfaces the people who genuinely hold a
skill. Use `relevant_candidate_ids` instead to pin exact ids by hand.

Three metrics, each answering a different question:

| Metric | Question | Notes |
|---|---|---|
| **MRR** | How soon does the *first* good result appear? | Position 1 → 1.00, position 2 → 0.50, position 4 → 0.25. Blind to everything after the first hit. |
| **MAP** | Are *all* the good results near the top? | Averages precision at every relevant hit. |
| **NDCG** | Same, with a log-scale position discount | Normalised against a perfect ranking, so 1.00 means ideal order. |

Sample run over the bundled accounting corpus (51 resumes, 10 queries, top 10):

```
  payroll processing               RR 1.000   AP 1.000   NDCG 1.000
  accounts payable clerk           RR 1.000   AP 1.000   NDCG 1.000
  candidate comfortable writing SQL queries
                                   RR 1.000   AP 0.900   NDCG 0.936
  cost accounting and variance analysis
                                   RR 0.500   AP 0.171   NDCG 0.357
  tax preparation specialist       RR 0.333   AP 0.233   NDCG 0.416

  MRR   0.767   MAP   0.557   NDCG  0.676
```

The spread is the interesting part. "budget planning and forecasting" scores
RR 1.000 but AP 0.307 — the very first result is right, yet the rest of the
page is mostly wrong, which is exactly the gap MRR alone would hide.

---

### Feature 10 — Re-index from the sidebar

Two explicit buttons under **Indexing** in the sidebar, both running the same
service as `index_resumes.py`, with live progress:

- **Index new resumes** — incremental. Parses only PDFs that are new or changed
  since the last run. Use this after dropping files into `data/resumes/`.
- **Force full rebuild** — discards the index and re-parses every resume. One
  Gemini call per resume, so reach for it only when you mean it.

Neither runs automatically. Startup never indexes.

---

## Project Structure

```
Module 2 - RAG/
├── pyproject.toml               # Shared dependency set for the whole module
├── uv.lock                      # Locked versions — uv sync installs from this
├── .python-version              # 3.11
├── .env                         # API keys (module root, NOT inside Hireflow/)
└── Hireflow/
    ├── api/
    │   └── main.py              # FastAPI backend (POST /index, /search, GET /status)
    ├── core/
    │   ├── indexing_service.py  # build / load / add — the one place indexing happens
    │   ├── index_store.py       # Persisted corpus + manifest (incremental hashing)
    │   ├── hybrid_indexer.py    # BM25 + Pinecone vector search with RRF fusion
    │   ├── vector_store.py      # Pinecone client wrapper
    │   ├── ingestion.py         # PDF loading + Gemini parsing -> LangChain Documents
    │   ├── parsing.py           # Gemini-based resume field extraction
    │   ├── candidate_evaluator.py # LLM qualitative assessment (no score)
    │   ├── filters.py           # Post-search filtering (skills/location/experience)
    │   └── retrieval_eval.py    # MRR / MAP / NDCG + the evaluation runner
    ├── utils/
    │   ├── schemas.py           # SearchQuery, Resume, CandidateEvaluation
    │   ├── config.py            # .env configuration loader
    │   ├── embeddings.py        # HuggingFace all-MiniLM-L6-v2
    │   └── utils.py             # Text processing, PDF loading, logging
    ├── streamlit/
    │   └── app.py               # Web interface
    ├── tests/
    │   ├── test_filters.py          # Filter function tests
    │   ├── test_hybrid_indexer.py   # RRF fusion and indexing tests
    │   ├── test_candidate_evaluator.py # LLM assessment + rule-based fallback
    │   ├── test_retrieval_eval.py      # MRR / MAP / NDCG worked examples
    │   ├── test_ingestion.py        # PDF loading tests
    │   └── test_indexing_service.py # Incremental/force indexing, startup cost
    ├── data/
    │   ├── resumes/             # Place PDF resumes here
    │   ├── jds/                 # Job description PDFs
    │   ├── hybrid_index/        # THE persisted index: corpus.pkl + manifest.json
    │   └── eval/                # Offline evaluation test cases
    ├── .streamlit/
    │   └── config.toml          # Disables the module watcher (see Troubleshooting)
    ├── index_resumes.py         # One-time indexing CLI — run this first
    ├── evaluate_retrieval.py    # Offline retrieval evaluation CLI
    ├── start_backend.py         # uvicorn launcher
    └── HireFlow_Architecture.md # Detailed architecture docs
```

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (gemini-2.5-flash-lite) |
| Vector DB | Pinecone (serverless, cosine, 384 dims) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Keyword Search | rank_bm25 (BM25Okapi) |
| Score Fusion | Reciprocal Rank Fusion (k=60) |
| Orchestration | LangChain |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| Retrieval Evaluation | MRR / MAP / NDCG (`core/retrieval_eval.py`) |
| Testing | pytest |
| Packaging | uv (Python 3.11) |

---

## Troubleshooting

**"Search returns nothing" / "No resume index found"**
The index hasn't been built. Run `uv run python index_resumes.py` from
`Hireflow/`, then reload the app. Check with `--status`.

**`ModuleNotFoundError: No module named 'torchvision'` on Streamlit startup**
Harmless noise, suppressed by `fileWatcherType = "none"`. Streamlit's source
watcher walks `__path__` on every module in `sys.modules`; `transformers`
registers ~270 lazily-loaded submodules, so the walk imports vision models like
`zoedepth` that want `torchvision`, and Streamlit logs each failure as a full
traceback. With the watcher off, `AppSession` never builds it at all.

**This setting is read from the directory you launch from, not the app's
directory.** A copy therefore lives in both places, and they must stay in sync:

```
Module 2 - RAG/.streamlit/config.toml     # streamlit run Hireflow/streamlit/app.py
Module 2 - RAG/Hireflow/.streamlit/config.toml   # streamlit run streamlit/app.py
```

If the tracebacks come back, you are launching from a third directory — check
with `streamlit config show | grep fileWatcherType`, which should print `none`.
The cost is no auto-reload on save, so restart the app after editing code.
(Installing `torchvision` would also fix it, but pulls in a large dependency for
no benefit here.)

**Dozens of `INFO: HTTP Request: ... huggingface.co ...` lines on startup**
Fixed, in two places. `get_logger` used to call `logging.basicConfig(level=INFO)`,
which sets the *root* logger to INFO and so turned on INFO logging for every
third-party library; the root handler now sits at WARNING while HireFlow's own
loggers opt in to INFO. Separately, `get_embeddings` now loads the model with
`local_files_only=True`, so a cached model is read from disk instead of being
revalidated against huggingface.co on every launch — verified at zero HTTP
requests. A cold cache still falls back to downloading once.

**Index looks stale after editing PDFs**
Incremental runs compare file content hashes, so an edited PDF is re-parsed
automatically. If the index itself is damaged, `--force` rebuilds from scratch.

**`pinecone_vector_count` is high but `indexed_resumes` is 0**
Pinecone holds vectors from an older run, but the local index was never built.
Run `index_resumes.py` — Pinecone upserts are keyed by `candidate_id`, so
re-indexing updates those vectors rather than duplicating them.
