# Authority Data

Synthetic authority-decision datasets for testing whether a model can follow
priority-ordered allow/disallow rules.

The Hugging Face dataset is organized as one repo with multiple configs:

| Config | Splits | Description |
| --- | --- | --- |
| `GeneralAuthorityV1` | `train`, `test` | General authority rules with deterministic query rendering. |
| `GeneralAuthorityV2` | `train`, `test` | General authority rules with LLM query paraphrases. |
| `ToolAuthorityV1` | `train`, `test` | Tool authority rules with deterministic query rendering. |
| `ToolAuthorityV2` | `train`, `test` | Tool authority rules with LLM query paraphrases. |


Each final row has this shape:

| Column | Description |
| --- | --- |
| `text` | Natural-language authority setting and query. |
| `label` | Gold decision, `Yes` or `No`. |
| `meta_data` | Structured query, paraphrase version, and query-meaning score for LLM paraphrases. |

## Layout

```text
authority-data/
  make_data.py
  hf_push.py
  src/
    make_base_authority_data.py
    make_paraphrase_data.py
    attribute_sampling.py
    category_sampling.py
    domains.py
    label_based_polarity_sampling.py
  data/
    base/
    paraphrase/
```

`make_data.py` builds structured base rows under `data/base/`.

`paraphrase_data.py` can replace the deterministic query rendering with
LLM-generated natural query paraphrases. The LLM receives only an instruction
and the structured query conditions, then a second LLM call scores whether the
natural query preserves the structured condition meaning with `0` or `1`.

## Build

```bash
python make_data.py
```

This creates structured base rows. Generate query paraphrase cache rows with
`paraphrase_data.py`; `hf_push.py` applies those cached paraphrases at push time.

## LLM Query Paraphrases

```bash
export OPENAI_API_KEY=...
python paraphrase_data.py --dry-run
python paraphrase_data.py --limit 10
```

The default LLM paraphrase version is `v2`, and the default model is
`gpt-5.4-mini`. The script records generation/evaluation usage and estimates
cost using input `$0.75`, cached input `$0.075`, and output `$4.50` per 1M
tokens.

LLM paraphrases are cached locally under `data/paraphrase_cache/<version>.jsonl`.
The cache key is only `input_text` plus `paraphrase_version`. `hf_push.py` loads
the base splits and replaces the original `Query` block with the cached
generated text at push time for V2 configs. By default, it pushes four Hugging
Face configs: `GeneralAuthorityV1`, `GeneralAuthorityV2`, `ToolAuthorityV1`,
and `ToolAuthorityV2`. The V2 configs are built from the matching V1 local base
directories.

Each cache row keeps only the fields needed to replace the query block:

```json
{
  "input_text": "{\"query_conditions\":{\"day\":\"wednesday\"}}",
  "paraphrase_version": "v2",
  "generated_text": "Action on Wednesday.",
  "evaluation_score": 1
}
```

The normal full workflow is:

```bash
python make_data.py
python paraphrase_data.py
python hf_push.py
```

For V2 uploads, `hf_push.py` pushes only rows that already have matching cache
entries and an evaluation score of `1`; rows with score `0` are skipped and
reported. Use `--strict-cache` when every base row must have a paraphrase before
pushing.

Generation uses a `tqdm` progress bar. Cache rows are appended as soon as each
example finishes, so an interrupted run resumes from the first missing cache
entry.

Useful smaller local run:

```bash
python make_data.py \
  --num-categories 1 \
  --max-k-pairs-per-category 2 \
  --num-samples-per-k-pair 2
```

## Push to Hugging Face

Default target repo: `leo-bjpark/authority`

```bash
python hf_push.py --dry-run
python hf_push.py
```

To remove old/wrong remote config folders such as `GeneralAuthority`,
`ToolAuthority`, `V1`, and `V2` before pushing:

```bash
python hf_push.py --delete-remote-configs --dry-run
python hf_push.py --delete-remote-configs
```

To push privately or with an explicit token:

```bash
python hf_push.py --private --token "$HF_TOKEN"
```

## Load

```python
from datasets import load_dataset

general_v1 = load_dataset("leo-bjpark/authority", "GeneralAuthorityV1")
general_v2 = load_dataset("leo-bjpark/authority", "GeneralAuthorityV2")
tool_v1 = load_dataset("leo-bjpark/authority", "ToolAuthorityV1")
tool_v2 = load_dataset("leo-bjpark/authority", "ToolAuthorityV2")

print(general_v2["train"][0])
```

## Generation Flow

1. Sample category combinations, or tool-specific recipient/information-type combinations.
2. Expand them into priority-ordered user authority rules.
3. Split each dataset config into `train` and `test`.
4. Render each structured authority setting into `text`.
5. Push the dataset configs to Hugging Face.
