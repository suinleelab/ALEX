# ResearchAgent: Iterative Research Idea Generation over Scientific Literature

[![Paper](https://img.shields.io/badge/arXiv-2404.07738-b31b1b)](https://arxiv.org/abs/2404.07738)
[![Python](https://img.shields.io/badge/Python-3.9%2B-orange)](https://www.python.org/downloads/release/python-390/)

🚀 **Welcome to the official repository of** [**ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models**](https://arxiv.org/abs/2404.07738)!

Authors: Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang

ResearchAgent leverages Large Language Models (LLMs) to help researchers rapidly ideate and refine research problems grounded in existing literature. Starting from a core scientific paper, the system iteratively proposes and improves problems, methods, and experiment designs using collaborating LLM-based reviewing agents that provide structured feedback across multiple dimensions.


## Overview

- Inputs: either (1) a built-in clinical trial paper context (`ist3`, `crash_2`, `sprint`, `accord`) or (2) Semantic Scholar paper IDs.
- Default mode: trial-paper-only (no external retrieval).
- Optional retrieval mode: fetch target papers, related references, and entities from Semantic Scholar/knowledge store.
- Problem Identification: generate a candidate research problem and rationale using LLMs.
- Problem Validation: obtain multi-criteria reviews and feedback from LLM reviewers (five metrics) in parallel.
- Iteration: refine the problem based on low-scoring aspects and repeat for a few rounds, keeping a concise history.


## Repository structure

- code/
	- main.py — entrypoint to run the end-to-end pipeline
	- knowledge/
		- store.py — lightweight knowledge store and entity retrieval
	- models/
		- openai.py — OpenAI Chat Completions wrapper with retries/timeouts
	- pipelines/
		- research_pipeline.py — orchestration of generate and validate iterations
		- agents/
			- base.py — shared prompt-formatting helpers
			- problem_identifier.py — generates/refines problems
			- problem_validator.py — reviews problems across 5 metrics in parallel
            - ...
	- utils/
		- s2.py — Semantic Scholar API helpers (papers, references, embeddings)
		- data_io.py — JSONL loading and ID utilities
		- formatting.py — small text utilities
- data/
	- papers.jsonl — input list of paper IDs
	- knowledge.jsonl — knowledge base (entities/co-occurrence)


## Running

### Trial-paper-only mode (default, no retrieval)

This mode uses only the selected trial paper context and does not retrieve references/entities.

```bash
export OPENAI_API_KEY=YOUR_KEY
python ./code/main.py \
  --trial-name ist3 \
  --model-name gpt-4o \
  --feature-name age \
  --pubmed-output ../../docs/results/ist3/researchagent_explanations.json
```

The `--pubmed-output` file is directly consumable by `pubmed_mechanism_validator.py`.
When `--pubmed-output` is set, the exporter automatically guarantees exactly 15 explanations (mechanisms) per run.

### Retrieval mode (optional)

Set your OpenAI key and run the pipeline:

```bash
export OPENAI_API_KEY=YOUR_KEY
python ./code/main.py \
	--data-path ./data/papers.jsonl \
	--knowledge-path ./data/knowledge.jsonl \
	--use-external-retrieval \
	--model-name gpt-4o
```


## Citation

If you use or build upon this project, please cite:

```bibtex
@inproceedings{Baek2025ResearchAgent,
  title={ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models},
  author={Jinheon Baek and Sujay Kumar Jauhar and Silviu Cucerzan and Sung Ju Hwang},
  booktitle={NAACL},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:269042844}
}
```
