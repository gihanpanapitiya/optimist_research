# Optimist Deep Research

An iterative deep-research agent for autonomous literature research, built on
LangGraph. It implements the methodology in [`plan.md`](plan.md): rather than
treating literature search as a single retrieval operation, it runs research
as an **iterative evidence-acquisition process** over an explicit, mutable
research state.

```
optimistresearch run "What are the recent developments in planning for AI agents?"
```

## What it does

Given a research question, the agent plans what must be established to answer
it, searches six literature sources, deduplicates what it finds into a
canonical corpus, screens and ranks candidates, reads the best ones to extract
structured evidence, synthesizes per subquestion and then globally, and
finally criticizes its own work — deciding whether to run another research
iteration or write the report.

The two properties that distinguish it from a search-and-summarize pipeline:

**The research plan is not fixed.** The critic can add subquestions the
literature revealed, split ones whose parts need separate evidence,
deprioritize dead ends, and rephrase questions that used the wrong
terminology. The decomposition is part of the evolving state, not an input to
it.

**The output is an evidence store, not a pile of summaries.** Each document
read yields structured claims, each with its support, location, type, and a
strength judgement. That is what lets the critic ask whether a claim is
actually established, and what makes every sentence in the report traceable to
a specific retrieved document.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[anthropic]'          # or '.[openai]', '.[google]', '.[ollama]'
pip install -e '.[anthropic,pdf]'      # add full-text PDF reading
cp .env.example .env                   # then edit
```

Set at minimum an API key for your model provider, and
`OPTIMISTRESEARCH_CONTACT_EMAIL` — it puts OpenAlex, Crossref and PubMed requests in
their polite pools, which have far better rate limits. Everything else is
optional: with no literature-source keys at all, arXiv, OpenAlex, Crossref and
PubMed all work, and OpenAlex supplies the citation graph.

```bash
optimistresearch sources    # which sources are configured and usable
optimistresearch config     # effective configuration
```

## Sources

| Source | Key | Citation graph | Best for |
| --- | --- | --- | --- |
| arXiv | — | no | Very recent CS/ML preprints |
| OpenAlex | — | both directions | Broad coverage, reliable DOIs, snowballing |
| Crossref | — | backward only | Published records; deposited reference lists |
| PubMed | optional | no | Biomedical and life sciences |
| Semantic Scholar | **effectively required** | both directions | Richest citation graph |
| Web (Tavily) | required | no | Grey literature, docs, blog posts |

arXiv asks for one request per three seconds, so it usually paces a search
round.

### Semantic Scholar is off by default

`semantic_scholar` is a three-way toggle: `auto` (the default), `on`, `off`.
Under `auto` it is used **only when `SEMANTIC_SCHOLAR_API_KEY` is set**.

That is not gatekeeping. The unauthenticated pool is shared per-IP and returns
429 under even light use, so an unkeyed Semantic Scholar spends
query-generation calls and search budget to return nothing. Leaving it off is
strictly better than leaving it on without a key.

```bash
optimistresearch sources                              # shows the resolved decision
optimistresearch run "..." --semantic-scholar on       # try the unkeyed pool anyway
optimistresearch run "..." --semantic-scholar off      # skip it even with a key
```

**Without a key, OpenAlex carries the citation graph.** It covers both
directions, needs no key, and the snowball source order is key-aware — so
snowballing keeps working, with one real caveat below.

### Citation coverage without a Semantic Scholar key

Forward snowballing (who cites this paper) runs on OpenAlex and works well.

Backward snowballing (what this paper cites) has a chain, tried in order until
one returns edges:

```
backward:  semantic_scholar (if keyed) → openalex → crossref
forward :  semantic_scholar (if keyed) → openalex
```

OpenAlex leads because every edge it returns is a full record with an
abstract, so those candidates are actually readable. Crossref backs it up
because publishers deposit reference lists there, and those are populated for
many records whose OpenAlex `referenced_works` is empty. Where both have data
Crossref returns *more* edges — 43 against 22 for one IEEE paper, 25 against
21 for a journal article — but roughly half of its entries carry no DOI, so
OpenAlex-first trades a little coverage for candidates that can be read.

**Crossref supplies the edge list; OpenAlex hydrates it.** A reference entry
is often just a DOI and a surname, which screening and reading cannot use. The
DOI-bearing entries are therefore batch-looked-up in OpenAlex (50 per request)
to recover titles, authors, venues and abstracts. Entries with a title but no
DOI survive as sparse candidates that screening can still judge; entries with
neither are dropped.

The residual gap is arXiv-only preprints. Their DOIs are registered with
DataCite rather than Crossref, so there is no deposited reference list, and if
OpenAlex's record also lacks `referenced_works` then nothing but Semantic
Scholar can reach their references. Chain-of-Thought is exactly this case.

### Two correctness notes

**Author names are folded before comparison.** OpenAlex writes
"Romera‐Paredes" with U+2010 HYPHEN while other sources use an ASCII hyphen;
accents vary too. Names and titles are Unicode-folded (NFKD, combining marks
stripped, dash variants unified) before matching, which fixed a live case
where a correct Nature resolution was rejected because the surnames compared
as "paredes" against "romera-paredes".

Relatedly, a **full title match is never vetoed by a disjoint author list**.
Sources render names too inconsistently — initials only, collective names,
transliterations — for an empty surname intersection to outweigh a matching
title. Author agreement is required only where the title evidence is weak (a
truncation or a prefix).

#### A correctness note on OpenAlex

OpenAlex has some arXiv DOIs attached to the wrong record — looking up
`10.48550/arXiv.2201.11903` (Chain-of-Thought Prompting) returns an unrelated
2022 paper. Snowballing from that would pull a stranger's references into the
corpus attributed to your seed, which is worse than not snowballing.

So every OpenAlex identifier lookup is **verified against the document's title
and authors** before use, with a punctuation-stripped title search as fallback.
A lookup that cannot be verified is refused and logged rather than trusted.

## Architecture

```
START → plan → generate_intents → generate_queries → search_all
                     ↑                                     ↓
                     │                            canonicalize_documents
                     │                                     ↓
                     │                              screen_and_rank ←──┐
                     │                                     ↓           │
                     │                              plan_snowball      │
                     │                                     ├── expand ─┘
                     │                                     ↓
                     │                      ⇉ read_document ⇉
                     │                                     ↓
                     │                 ⇉ synthesize_subquestion ⇉
                     │                                     ↓
                     │                             global_synthesis
                     │                                     ↓
                     └─── apply_critique ←── critique_research ──→ write_report → END
```

`⇉` marks a `Send` fan-out. Two loops with different jobs: the inner one
(snowball ⟲ screening) expands the citation neighbourhood and stops on a
saturation criterion; the outer one (critique ⟲ planning) is the adaptive
research loop that can rewrite the plan itself.

### Where the parallelism is

LLM-heavy stages — document reading, per-subquestion synthesis — fan out with
`Send`, so each gets its own traced node and one malformed extraction cannot
take down the others. HTTP search runs concurrently *inside* `search_all` with
a semaphore instead: each source already enforces its own rate limit, and one
graph node per `(source, query)` pair would bloat the state for no benefit.

### The hierarchy

```
Research question → subquestion → search intent → retrieval query → source
```

The two middle levels are separate on purpose. A **subquestion** says what we
need to *know*; a **search intent** says what documents would close that gap.
Keeping them apart means query phrasings can be rewritten freely without
disturbing the research objective they serve.

Routing uses **Option 3** from the methodology: select the relevant sources
first, then generate a query specialized for each. One query string cannot
serve arXiv (literal keyword matching, ANDed terms), Semantic Scholar
(semantic ranking over natural language) and PubMed (controlled vocabulary) at
once.

## Module map

| Path | Role |
| --- | --- |
| `optimistresearch.yaml` | The settings you edit; `.env` holds only secrets |
| `state.py` | The research state and its LangGraph reducers — the spine |
| `schemas.py` | Structured-output schemas for the LLM nodes |
| `config.py` | Budgets, ranking weights, per-role model routing |
| `llm.py` | Provider-agnostic model factory (`init_chat_model`) |
| `graph.py` | Graph assembly |
| `nodes/planner.py` | Decomposition into subquestions |
| `nodes/intents.py` | Subquestion → search intents |
| `nodes/queries.py` | Source routing + per-source query generation |
| `nodes/search.py` | Concurrent multi-source search, canonicalization |
| `nodes/screen.py` | Relevance screening, ranking, reading selection |
| `nodes/snowball.py` | Backward/forward citation expansion, saturation |
| `nodes/reader.py` | Parallel structured evidence extraction |
| `nodes/synthesis.py` | Per-subquestion and global synthesis |
| `nodes/critic.py` | Reflection, plan mutation, termination |
| `nodes/report.py` | Section-by-section report generation |
| `store/canonicalize.py` | The deduplication hierarchy |
| `store/evidence.py` | Claim clustering, support and diversity metrics |
| `store/corpus.py` | On-disk research corpus |
| `fulltext.py` | PDF fetch + section-aware extraction (gated) |
| `mcp_server.py` | MCP server for coding agents (`optimistresearch-mcp`) |

## Output

```
research/<slug>/
├── research_plan.md          the subquestion tree, intents, gaps
├── research_state.json       complete state; `--resume` reads this
├── titles.md                 human-readable document index
├── sources/doc_0001/
│   ├── metadata.json         canonical record + full provenance
│   ├── notes.md              reading, extracted evidence, abstract
│   └── paper.pdf             only with --full-text
├── subtopics/<subquestion>.md
├── evidence/
│   ├── claims.md             the evidence store, by claim cluster
│   └── contradictions.md
└── final_report.md
```

The state is checkpointed after every graph step, so an interrupted run leaves
a usable partial corpus.

## Configuration

Settings live in **`optimistresearch.yaml`**, which is the file to commit. Secrets
stay in `.env`, which is gitignored.

```yaml
models:
  # Judgement roles: once per iteration, so a capable model is cheap here.
  planner: openai:gpt-5.4
  critic: openai:gpt-5.4
  synthesizer: openai:gpt-5.4
  writer: openai:gpt-5.4
  # Per-document roles: these dominate spend, so they step down a tier.
  intent: openai:gpt-5.4-mini
  query: openai:gpt-5.4-mini
  reader: openai:gpt-5.4-mini
  screener: openai:gpt-5.4-nano    # highest volume of all

enabled_sources: [arxiv, openalex, crossref]

budget:
  max_documents_read: 40
  max_research_iterations: 3
```

Per-role routing is the main cost control available: screening and reading run
once per *document*, planning and criticism once per *iteration*. Any
`provider:model` spec `init_chat_model` understands works, so providers can be
mixed across roles.

### Precedence

Lowest first, so a committed file can always be overridden for one run:

1. defaults in `optimistresearch/config.py`
2. `optimistresearch.yaml`
3. `.env` — API keys, and machine-local overrides
4. environment variables, including those the CLI sets from its flags

```bash
optimistresearch config                            # what is in effect, and from which files
optimistresearch run "..." --max-documents 20      # beats the committed YAML
OPTIMISTRESEARCH_BUDGET__MAX_DOCUMENTS_READ=20 ... # the same, as an env var
```

Both files are found by walking up from the working directory and then falling
back to the project root, nearest winning — so a notebook started anywhere
still finds the configuration. YAML files merge **key by key**, so a
`optimistresearch.yaml` in a subdirectory can change one budget value without
restating the block. `OPTIMISTRESEARCH_CONFIG_FILE` pins an explicit file.

Environment variables still work for everything (`OPTIMISTRESEARCH_` prefix, `__`
for nesting) and are the right tool for a single override. They are a poor
tool for a whole configuration: a key repeated in a `.env` silently takes its
last value, which is how an edit meant to raise a concurrency limit left it at
the lower value further down the file. A duplicated YAML key is a visible
structural error instead.

### Running fully locally with Ollama

Supported, and it needs no API keys at all — neither for the models nor for
arXiv, OpenAlex, Crossref and PubMed.

```bash
pip install -e '.[ollama]'
```

```yaml
# optimistresearch.yaml -- point every role at the SAME model
models:
  planner: ollama:gemma4:latest
  intent: ollama:gemma4:latest
  query: ollama:gemma4:latest
  screener: ollama:gemma4:latest
  reader: ollama:gemma4:latest
  synthesizer: ollama:gemma4:latest
  critic: ollama:gemma4:latest
  writer: ollama:gemma4:latest

ollama_num_ctx: 32768
max_concurrent_readers: 2
```

Three things decide whether a local model works here, and parameter count is
the least of them.

**Structured output is free.** `with_structured_output` resolves to Ollama's
`json_schema` mode, which constrains decoding to the schema. Output always
parses, so the question is never malformed JSON — it is whether the *content*
is any good.

**Context is the trap.** Ollama defaults to a 4096-token window whatever the
model supports, and truncates silently rather than erroring. A screening
prompt carries ten abstracts, so the default would quietly wreck the run.
`ollama_num_ctx` is passed on every call for this reason.

**Use one model for every role.** Per-role routing is great against hosted
APIs but counterproductive locally: switching models makes Ollama unload and
reload, and that dominates the runtime. Set every role to the same model and
keep `max_concurrent_readers` low, since Ollama serializes anyway and a wide
fan-out only queues requests.

#### Tuning for local inference

Expect roughly 30–45 minutes for a small run (8 documents, 2 iterations) at
12–45 seconds per call, versus a couple of minutes against a hosted API. Two
settings dominate that, and both default to values tuned for hosted models:

```yaml
budget:
  screen_batch_size: 5              # default 10
  max_snowball_seeds: 2             # default 6
  max_snowball_edges_per_seed: 15   # default 40
```

Screening dominates wall-clock, because it is paid per candidate and the call
must emit a score per document per subquestion. Smaller batches are faster to
generate and keep a small model from losing track of which index it is
scoring.

Snowball breadth needs to track the reading budget. On a measured run with an
8-document budget, the defaults produced **120 new candidates from a single
round** — every one of which then had to be screened before it could be
discarded. That is reasonable when you will read a hundred documents and
wasteful when you will read eight.

Budget ceilings are limits, not targets — the critic may stop well before any
is reached:

```bash
OPTIMISTRESEARCH_BUDGET__MAX_DOCUMENTS_READ=100
OPTIMISTRESEARCH_BUDGET__MAX_SEARCH_OPERATIONS=60
OPTIMISTRESEARCH_BUDGET__MAX_SNOWBALL_ROUNDS=3
OPTIMISTRESEARCH_BUDGET__MAX_RESEARCH_ITERATIONS=4
```

Number of papers retrieved is deliberately *not* a stopping criterion.

### Reading mode

Default is abstract-only: evidence is extracted from title, abstract and
metadata. This is fast and cheap enough to iterate on the graph logic, and the
reader is told to cap claim strength accordingly since it cannot verify
experimental setups. `--full-text` fetches PDFs and extracts section-aware
text, so `location` fields point at real sections.

```bash
pip install -e '.[pdf]'
optimistresearch run "..." --full-text
```

## Programmatic use

Call the agent from a Python script, or wrap it as a tool in your own agent:

```python
import asyncio

from optimistresearch.config import configure
from optimistresearch.runner import run_research, summarize_run

# Optional per-run overrides, the counterpart of the CLI flags. Anything not
# given comes from optimistresearch.yaml, .env and OPTIMISTRESEARCH_* as usual.
configure(
    budget={"max_documents_read": 20, "max_research_iterations": 2},
    enabled_sources="openalex,arxiv",
    read_full_text=False,
    output_dir="research",
)

seen = 0

def on_progress(event: str, snapshot: dict) -> None:
    """Print new research-log lines; called after every graph step."""
    global seen
    log = snapshot.get("log") or []
    for line in log[seen:]:
        print(line)
    seen = len(log)

async def main() -> None:
    state = await run_research(
        "How do LLM agents recover from planning failures?",
        scope_notes="Focus on empirical evaluations since 2023.",
        on_progress=on_progress,
    )
    summary = summarize_run(state)
    print(summary["report_path"], summary["documents_read"], "documents read")

    print(state["final_report"][:500])
    for item in state["evidence"]:
        doc = state["documents"][item.doc_id]
        print(item.subquestion_id, item.strength, item.claim, "--", doc.title, doc.doi)

asyncio.run(main())
```

`run_research` loads `.env` itself, using the same search as the settings
(upwards from the working directory, then the project root), so provider keys
such as `OPENAI_API_KEY` are picked up wherever the script runs. Keys already
in the environment win. In a notebook, drop `asyncio.run` and `await
run_research(...)` directly.

The returned state is a dict. The useful keys:

| Key | Contents |
| --- | --- |
| `final_report` | The report, as Markdown |
| `evidence` | List of `Evidence`: `claim`, `support`, `doc_id`, `subquestion_id`, `evidence_type`, `strength` (0–1), `verbatim_quote` |
| `documents` | `doc_id` → `Document`: `title`, `authors`, `year`, `venue`, `doi`, `url`, `abstract`, `status` |
| `subquestions` | Research-plan id (`RQ1`, ...) → `Subquestion`, with its `coverage` |
| `contradictions` | Conflicting claims the critic found |
| `terminated_because` | Why the run stopped |

`summarize_run(state)` gives the headline numbers and the `report_path` and
`corpus_root` on disk; the files there are listed under [Output](#output).
Settings are process-wide, so run one research job per process at a time.

## Use from coding agents (MCP)

`optimistresearch-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io)
server, so coding agents like Claude Code, Cursor or Codex can hand a
literature question to Optimist Deep Research and read the report back.

```bash
pip install -e '.[mcp,anthropic]'
claude mcp add optimistresearch -- optimistresearch-mcp
```

Or, without installing into the agent's environment, let `uv` run it from a
checkout:

```bash
claude mcp add optimistresearch -- \
  uv run --project /path/to/optimist_research --extra mcp --extra anthropic optimistresearch-mcp
```

Other clients take the same command in their JSON config:

```json
{
  "mcpServers": {
    "optimistresearch": {
      "command": "optimistresearch-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPTIMISTRESEARCH_CONTACT_EMAIL": "you@example.org",
        "OPTIMISTRESEARCH_OUTPUT_DIR": "/path/to/research"
      }
    }
  }
}
```

**API keys.** The server reads the same `.env` as the CLI: it searches upwards
from the directory the agent starts it in, then the checkout the package was
installed from with `pip install -e` or `uv run --project`. With a regular,
non-editable install there is no checkout to fall back to, so pass the keys in
the client's `env` block as above (with Claude Code, `claude mcp add -e
OPENAI_API_KEY=... optimistresearch -- optimistresearch-mcp`). Model routing and
budgets come from `optimistresearch.yaml`, found the same way, or from
`OPTIMISTRESEARCH_CONFIG_FILE`.

Then ask the agent in plain language:

> Use optimistresearch to research how LLM agents recover from planning
> failures, limited to 20 documents. When it's done, summarise the main
> findings and list the five strongest supporting papers.

| Tool | What it does |
| --- | --- |
| `start_research` | Start a run in the background; returns a `run_id` at once |
| `research_status` | Status, the tail of the research log, and headline numbers when done |
| `get_report` | The final report, as Markdown |
| `get_corpus_file` | Any text file in the corpus, e.g. `evidence/claims.md` |
| `list_research` | Runs in the output directory, newest first |
| `cancel_research` | Stop a run; the partial corpus stays and can be resumed |
| `list_sources` | Which literature sources are usable |

`start_research` takes the question plus optional `scope_notes`,
`max_documents`, `max_iterations`, `full_text` and `resume`.

A run takes minutes, much longer than an agent should block on one tool call,
so `start_research` returns immediately and the agent checks `research_status`
until the run reports `completed` (the server's instructions tell it to check
about once a minute). Only one run is active at a time. The `run_id` is the
corpus directory name, and status and reports are read from disk, so runs from
earlier sessions, or ones the CLI wrote, can be looked up the same way.

The agent starts the server in the project you are working in, so the default
`research/` output lands there. Set `OPTIMISTRESEARCH_OUTPUT_DIR` to keep
corpora in one place. If the agent's session ends mid-run the run stops with
it, but the checkpointed corpus survives: `start_research` with `resume: true`
picks it up again.

## Open questions from the methodology

`plan.md` raises several questions it leaves to testing. Where the code takes a
position, it is a testable one:

**Does subquestion decomposition find more than a single query would?** The
planner is instructed toward the *minimum* set of meaningful subquestions, and
per-intent yield is recorded (`relevant_found` against `documents_found`), so
decomposition's contribution is measurable rather than assumed. Running with
`max_subquestions=1` gives the baseline to compare against.

**Should the report follow the subquestion structure?** No — global synthesis
derives the outline from the findings, and the prompt says so explicitly. The
initial plan guides acquisition; it does not determine the report's structure.

**Would a concept graph make the report more coherent?** Not implemented. The
evidence store is currently claim-clustered by exact text, with semantic
reconciliation left to the synthesizer. A concept-graph layer would slot in at
`store/evidence.py` as a richer clustering over the same data.

## Development

```bash
pip install -e '.[anthropic,dev,pdf]'
pytest                              # 250 tests, fully offline
ruff check optimistresearch tests scripts
python scripts/smoke_sources.py     # live source check, makes real requests
python scripts/check_models.py openai:gpt-5.4-mini   # can a model cope?
```

`pytest` covers the deterministic machinery — the dedup hierarchy, state
reducers under fan-in, ranking and reading allocation, snowball saturation,
source parsing against recorded payloads, corpus round-tripping — plus the
whole graph end to end against a fake model and fake sources. It does not test
model output quality.

`scripts/check_models.py` runs a candidate model against the real schemas and
prompts, flagging the failure modes that matter: keeping a deliberately
off-topic paper, returning `relevant=False` on a relevant one, or copying
search intents out of the prompt's own examples. It also caught a model
returning `"RQ1: How do agents..."` where a bare `RQ1` was expected — which
would have scored every document zero. Use it before trusting a new model.

Run `scripts/smoke_sources.py` after touching a source adapter. The scholarly
APIs reject an entire request rather than ignoring one bad field, so a single
wrong field name costs a whole source while every offline test still passes —
which is exactly how three real bugs in these adapters were found.

## Citation

If you use Optimist Deep Research in your work, please cite the paper:

```bibtex
@misc{panapitiya2026optimistdeepresearchiterative,
      title={Optimist Deep Research: An Iterative, State-Centric Agent for Autonomous Literature Research},
      author={Gihan Panapitiya},
      year={2026},
      eprint={2610.23299},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.23299},
}
```

## Disclaimer

Optimist Deep Research is a research aid, not a source of truth. Its reports
are generated by language models and can contain errors: misread or
misattributed claims, citations that do not support the sentence they are
attached to, and gaps in coverage when relevant work was not retrieved or not
read. Check every claim against the cited sources before relying on it. The
reports are not professional, medical, legal or financial advice.

You are responsible for how you use the software and its output, including
complying with the terms of service and rate limits of the APIs it calls
(OpenAlex, Crossref, PubMed, arXiv, Semantic Scholar, web search providers
and LLM providers) and with the copyright and licensing terms of any full
text it retrieves.

The software is provided "as is", without warranty of any kind, and the
authors accept no liability for any consequences of its use. See sections 7
and 8 of the [license](LICENSE).

## License

Copyright 2026 Gihan Panapitiya.

Licensed under the [Apache License, Version 2.0](LICENSE). "Optimist" and
"Optimist Deep Research" are names of this project; the license does not grant
permission to use them, beyond describing where the work came from (section 6).
