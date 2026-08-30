# TechJam Conversational E-Commerce Search Challenge

Public repository: `<replace_with_your_github_repo_url>`

This repository contains a submission for the TechJam Conversational E-Commerce Search Challenge. The solution is built around a multi-turn shopping agent with a boundary-first approach: it can tell when the customer has no clear preference, avoid making bad guesses, and not ask too many extra questions.

## Project Overview

The main focus of this project is good behavior on boundary sessions, where the customer may have no preference for the requested attribute. That means the agent is more careful about when to ask, when to recommend, and when to stop forcing a rule that does not fit.

This repo includes the full submission stack:

- `starter/agent.py` and `agent.py` for the agent entry point
- `evaluator/local_evaluator.py` for local scoring
- `data/public_set.jsonl` for the public development sessions
- `data/catalog.jsonl` for the frozen catalog used by the agent
- `docs/` for the challenge rules, API contract, and scoring details

The solution can be run locally with the public evaluator and the included data files. We treat boundary handling as a main goal, not a side case.

## Boundary-First Strategy

The main selling point is how it acts when the user's preference is weak, missing, or changes.

Key ideas:

- Detect boundary behavior early instead of making a guess.
- Ask fewer, useful questions when the customer has no clear preference.
- Give broad but valid recommendations instead of narrowing too much.
- Keep the chat steady when the user changes their mind.

This matters most for:

- Boundary sessions, where the user may really have no preference.
- Intent override sessions, where old preferences no longer fit.
- Browsing sessions, where too many questions can slow things down.

## Setup and Installation Instructions

### Requirements

- Python 3.10 or later
- Git
- Git LFS, if you need to work with the large catalog file in this repository

### Install

1. Clone the repository.
2. Ensure the `data/catalog.jsonl` file is present.
3. If needed, install any dependencies used by your agent implementation.

If your solution only uses the standard library, you do not need extra packages.

### Run the Local Evaluator

```bash
python -m evaluator.local_evaluator
```

This writes local results to `results.json`.

## Steps to Reproduce Results

1. Open the repository in a clean environment.
2. Verify that `data/catalog.jsonl` and `data/public_set.jsonl` are available.
3. Run the local evaluator:

```bash
python -m evaluator.local_evaluator
```

4. Review the resulting metrics in `results.json`.
5. Edit `starter/agent.py` or your custom agent files and rerun the evaluator to compare performance.

If you use external models or APIs, note any needed environment variables and network needs here.

## Why This Solution Stands Out

Many basic approaches ask too much or guess too hard when the customer is vague. This submission focuses on boundary-aware behavior so the agent can:

- stay useful even when there is no preference to find,
- avoid making up rules that were never given,
- keep good recommendations while asking fewer extra questions,
- stay stable when the user pushes back or changes direction.

## Solution Limitations

This solution is simple and has a few limits:

- It depends on the quality and completeness of the catalog and public session data.
- It may still miss some subtle boundary and intent-override cases.
- If the agent uses a weak baseline or simple retrieval method, performance can level off on vague sessions.
- Any LLM-based or API-based strategy may be affected by latency, token cost, or network availability.

### What I Would Improve With More Time

- Add a stronger retrieval and reranking pipeline.
- Improve boundary detection and no-preference handling.
- Improve intent-override detection and chat-state tracking.
- Make clarification questions better for vague or no-preference cases.
- Add better fallback logic for low-confidence turns.
- Measure and improve speed, cost, and reply consistency.

## Team Member Contributions

Team members and workload split:

- Andrew: boundary handling, fallback logic, and reply rules
- Austin: search pipeline, catalog lookup, and ranking work
- Felicia: testing, runs, and score tracking
- Valencia: agent code, integration, and cleanup
- Made: docs, submission files, and README polish

If you need a shorter format for submission, you can use this template:

| Team Member | Contributions |
| --- | --- |
| Andrew | Boundary handling, fallback logic, and reply rules |
| Austin | Search pipeline, catalog lookup, and ranking work |
| Felicia | Testing, runs, and score tracking |
| Valencia | Agent code, integration, and cleanup |
| Made | Docs, submission files, and README polish |

## Challenge Summary

Build an AI shopping agent that asks good follow-up questions and recommends the customer's hidden target product within 10 turns, with extra care for boundary behavior.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior, and this project is tuned to do well on boundary sessions.

## Download the Catalog

If you need to restore the catalog from the compressed release file, download `catalog.jsonl.gz` from the GitHub Release for this repo, then decompress it into `data/catalog.jsonl`.

Verify the downloaded file using the published `SHA256SUMS` file.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is part of the `Technical Execution` assessment. It is not a separate score and does not cover the whole `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Evaluation Focus

In our final submission, the main story is boundary-aware behavior. If you are reviewing this repo, pay special attention to:

- how the agent acts when the user has no preference,
- how it avoids asking the same kind of question again,
- how it recovers when a stated preference no longer fits,
- and how it keeps good recommendations when things are unclear.

## Model Choice and Cost

Teams may use any legal LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token use, and speed must be shared. Token use is a practical measure, not part of the core score. The organizer does not provide or pay for model API credits; teams pay for any outside services they use.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
data/catalog.jsonl                frozen product catalog
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
