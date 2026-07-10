# Authority Data

Synthetic authority-decision datasets for testing whether a model can follow
priority-ordered allow/disallow rules.

The Hugging Face dataset is organized as one repo with multiple configs:

| Config | Splits | Description |
| --- | --- | --- |
| `GeneralAuthority` | `train`, `test` | Authority rules over request attributes. |
| `ToolAuthority` | `train`, `test` | Tool-dependent authority rules over `tool`, `information_action`, and `information_type`. |


Each final row has this shape:

| Column | Description |
| --- | --- |
| `id` | Split-local row id. |
| `text` | Natural-language authority setting and query. |
| `AttributeCombination` | Structured query attributes. |
| `Label` | Gold decision, `Yes` or `No`. |
| `metadata` | Categories, priority order, conflict flag, base authority setting, source ids, and prompt/paraphrase versions. |

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

`make_data.py` builds structured base rows first, then renders the final
paraphrased JSONL files under `data/paraphrase/`.

## Build

```bash
python make_data.py
```

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

To push privately or with an explicit token:

```bash
python hf_push.py --private --token "$HF_TOKEN"
```

## Load

```python
from datasets import load_dataset

general = load_dataset("leo-bjpark/authority", "GeneralAuthority")
tool = load_dataset("leo-bjpark/authority", "ToolAuthority")

print(general["train"][0])
```

## Generation Flow

1. Sample category pairs or tool/action/information-type combinations.
2. Expand them into priority-ordered user authority rules.
3. Split each dataset config into `train` and `test`.
4. Render each structured authority setting into `text`.
5. Push the dataset configs to Hugging Face.
