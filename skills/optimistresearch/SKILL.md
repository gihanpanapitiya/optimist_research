---
name: optimistresearch
description: Run Optimist Deep Research, an iterative literature-research agent, on a research question and report back what the literature says, with every claim traced to a retrieved paper. Use when the user asks for a literature review, a survey of papers, "what does the research say about X", prior work, related work, state of the art, or evidence for a scientific or technical claim — or names optimistresearch / Optimist Deep Research. Not for general web lookups or questions answerable without academic sources.
argument-hint: "<research question> [--max-documents N] [--full-text]"
allowed-tools: Bash, Read, Grep, Glob
---

# Optimist Deep Research

Optimist Deep Research plans subquestions, searches arXiv, OpenAlex, Crossref,
PubMed (and Semantic Scholar / the web when keyed), snowballs citations, reads
the best papers into a structured evidence store, critiques its own coverage,
and writes a cited report. A run takes minutes against a hosted model, so it
runs in the background and you read the results from disk.

The request: `$ARGUMENTS`

## 1. Find the command

```bash
command -v optimistresearch
```

If it is missing, use `uvx`, which needs no install step (pick the extra that
matches the provider in `optimistresearch.yaml`; `openai` is the default):

```bash
uvx --from "optimistresearch[openai] @ git+https://github.com/gihanpanapitiya/optimist_research" optimistresearch
```

Write `OR` below for whichever form works. If neither `optimistresearch` nor
`uvx` exists, tell the user to run
`pip install "optimistresearch[openai] @ git+https://github.com/gihanpanapitiya/optimist_research"`
and stop.

## 2. Check configuration

```bash
OR sources
```

- **No model API key / no config found:** the user needs to run `OR init ~`
  once and put their provider key (e.g. `OPENAI_API_KEY`) and
  `OPTIMISTRESEARCH_CONTACT_EMAIL` in the `.env` it writes. Ask them to do this
  — never ask them to paste a key into the chat, and never print `.env`.
- **No sources usable:** report the reasons the command printed and stop.
- Semantic Scholar showing as off without a key is normal; OpenAlex carries
  the citation graph.

## 3. Shape the run

Turn the request into a focused **question** plus optional **scope notes**
(time range, domains, study types, what to exclude). If the request is too
vague to research (e.g. "AI"), ask one clarifying question first.

Flags for `OR run "<question>"`:

| Flag | Use |
| --- | --- |
| `--scope "<notes>"` | Constraints for the planner |
| `--max-documents N` | Papers read in depth (default from YAML, usually 40). Use ~10–15 for a quick look |
| `--max-iterations N` | Research iterations (default 3). 1–2 for a quick look |
| `--full-text` | Read PDFs instead of abstracts (slower; needs the `pdf` extra) |
| `--sources a,b` | Restrict sources, e.g. `pubmed,openalex` for biomedical |
| `--resume` | Continue an interrupted run of the same question |
| `-o DIR` | Output directory (default `research/`) |

Honour any budget the user states. If they state none, use the defaults.

## 4. Run it in the background

Start it with the Bash tool's `run_in_background: true`, logging to a file:

```bash
OR run "<question>" --scope "<notes>" [flags] 2>&1 | tee "<scratchpad>/optimistresearch.log"
```

Tell the user it has started and roughly how long it may take (a few minutes
hosted; 30+ minutes on a local Ollama model). You are notified when it exits —
do not poll in a loop. If the user asks for progress meanwhile, show the tail
of the log, which prints one line per research step.

When it finishes, the log ends with headline numbers plus `Report:` and
`Corpus:` paths. A non-zero exit means it failed: show the last lines of the
log and the likely cause (missing key, rate limit, model error). An
interrupted run keeps its corpus; offer `--resume`.

## 5. Read the results

In the corpus directory (`research/<slug>/`):

| File | Contains |
| --- | --- |
| `final_report.md` | The cited report — read this first |
| `evidence/claims.md` | Every claim grouped by subquestion, with support count, strength (0–1), quote and full citation |
| `evidence/contradictions.md` | Conflicts the critic found |
| `research_plan.md` | Subquestions, coverage, remaining gaps |
| `titles.md` | Index of all documents with ids |
| `sources/doc_NNNN/metadata.json` | Title, authors, year, venue, DOI, URL |

## 6. Answer the user

Answer **their** question from the report and evidence; do not paste the
whole report. Include:

- The main findings, each tied to the paper(s) behind it (author, year, title,
  DOI or URL from `metadata.json`).
- How strong the evidence is: note claims with low strength or a single
  source, and any contradictions.
- Gaps the run reported (from `research_plan.md` and the termination reason),
  and whether reading was abstract-only.
- The path to `final_report.md` and the corpus.

Do not add findings or citations that are not in the corpus. The report is
model-generated; say that key claims should be checked against the cited
papers before being relied on.

For follow-ups, grep the existing corpus before starting a new run. To dig
deeper on the same question, rerun with `--resume` and a larger budget.
