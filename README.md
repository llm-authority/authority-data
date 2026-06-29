# Authority Data

Dataset utilities and prepared JSONL files for authority-decision experiments.

The import package is `authority_data`. It contains dataset exporters,
benchmark normalizers, prompt renderers, and prepared local data files. Local
maintenance commands live under `scripts/`.

## Install

For local development:

```bash
pip install -e ".[hf,notebooks]"
```

The base package has no required runtime dependencies. Install the `hf` extra
when you want to load or push datasets with Hugging Face `datasets`.

## Data Layout

```text
authority_data/
  authority.py          # authority dataset exporter
  benchmarks.py         # AgentDojo / InjecAgent benchmark exporters
  prompts.py            # shared prompt_v1 / prompt_v2 renderers

  data/
    authority/
      permission/
      prohibition/
      permission_and_prohibition/
    benchmarks/
      agentdojo/
      injecagent/

scripts/
  make_authority_data.py  # make local JSONL files
  hf_push.py              # push prepared files to Hugging Face Hub
```

Each dataset config is stored as `train.jsonl` and `test.jsonl`.

## Quick Use

```python
from importlib.resources import files
from datasets import load_dataset

config_dir = files("authority_data").joinpath("data", "authority", "permission")
dataset = load_dataset(
    "json",
    data_files={
        "train": str(config_dir / "train.jsonl"),
        "test": str(config_dir / "test.jsonl"),
    },
)
```

## Local Explorer

Start the local data explorer:

```bash
npm run dev
```

Then open the URL printed by the command. The explorer reads JSONL files from
`authority_data/data/`.

## Commands

Make all local JSONL files:

```bash
python scripts/make_authority_data.py
```

Make only the synthetic authority data:

```bash
python scripts/make_authority_data.py --skip-agentdojo --skip-injecagent
```

Make normalized benchmark data:

```bash
python scripts/make_authority_data.py --skip-authority
```

Push prepared files to Hugging Face:

```bash
python scripts/hf_push.py --repo-id leo-bjpark/authorization_data_v1
```

Upstream source repositories are cloned under `authority_data/sources/` by the
preparation commands and are intentionally ignored by git.

## Notebooks

Tutorial notebooks live under `notebooks/tutorials/`:

- `authority.ipynb`: load and inspect the authority dataset configs.
- `benchmarks.ipynb`: load and inspect the AgentDojo/InjecAgent configs.
