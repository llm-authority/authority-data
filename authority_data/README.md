# Authority Data Package

`authority_data` is the importable package for the repository. The root
`README.md` has the full install and usage guide.

Key modules:

```text
authority.py      # authority decision dataset generation
benchmarks.py     # AgentDojo / InjecAgent normalization
prompts.py        # shared prompt_v1 / prompt_v2 renderers
```

Prepared JSONL files live under `authority_data/data/`. Upstream clones created
by preparation commands live under `authority_data/sources/` and are ignored by
git. Repository-level command scripts live in `../scripts/`.
