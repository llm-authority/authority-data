# Authority Data

Synthetic authority-decision datasets for evaluating whether a model can follow
priority-ordered allow/disallow rules.

Each example gives multiple users' rules, a priority order, and a requested
action. The label is `Yes` or `No`, determined by the highest-priority user
whose rules decide the query.

## Configs

| Config | Query style | Main focus | Train | Test | Total |
| --- | --- | --- | ---: | ---: | ---: |
| `GeneralAuthorityV1` | Deterministic bullets | General rules, mixed conflict/non-conflict | 500 | 1,500 | 2,000 |
| `GeneralAuthorityV2` | LLM paraphrase | Same as V1 with natural queries | 491 | 1,468 | 1,959 |
| `GeneralAuthorityV3` | Deterministic bullets | Conflict-only, many-user stress test | 500 | 1,500 | 2,000 |
| `GeneralAuthorityV4` | Deterministic bullets | First applicable user, higher-priority non-matches | 500 | 1,500 | 2,000 |
| `GeneralAuthorityV5` | Deterministic bullets | V1-style all-applicable users matched to V4 user/rule counts | 500 | 1,500 | 2,000 |
| `ToolAuthorityV1` | Deterministic bullets | Tool-use rules, mixed conflict/non-conflict | 1,000 | 3,000 | 4,000 |
| `ToolAuthorityV2` | LLM paraphrase | Same as V1 with natural queries | 991 | 2,893 | 3,884 |
| `ToolAuthorityV3` | Deterministic bullets | Conflict-only, many-user stress test | 1,000 | 3,000 | 4,000 |
| `ToolAuthorityV4` | Deterministic bullets | First applicable user, higher-priority non-matches | 1,000 | 3,000 | 4,000 |
| `ToolAuthorityV5` | Deterministic bullets | V1-style all-applicable users matched to V4 user/rule counts | 1,000 | 3,000 | 4,000 |

V2 has fewer rows because only semantically verified paraphrases are kept. V3
is not paraphrased.

## Split Design

V1/V2 use a generalization-oriented split:

| Family | Split | Rule types | Tools | Users |
| --- | --- | --- | --- | --- |
| GeneralAuthority | `train` | `date`, `day`, `time` | N/A | 1-3 |
| GeneralAuthority | `test` | `date`, `day`, `month`, `time`, `year` | N/A | 1-5 |
| ToolAuthority | `train` | `information type`, `purpose`, `recipient` | `send_email`, `send_text_message`, `submit_online_form` | 1-3 |
| ToolAuthority | `test` | `day`, `information type`, `purpose`, `recipient`, `time` | `make_phone_call`, `send_voice_message`, `start_live_chat` | 1-5 |

V3 keeps V1-style deterministic queries but makes every example a conflict
case. The main user receives the gold allow/disallow context; a controlled
fraction of lower-priority users receives the same query context with the
decision flipped. V3 keeps user-count stress while capping total prompt rules
at 1,000 by default.

V4 is derived from the V1 splits. The deciding user is placed at a random
priority depth. Higher-priority users keep explicit allow/disallow rules, but
their rules omit the queried values, so they are not applicable to the query.
The label is decided by the first priority-ordered user whose rules explicitly
match all queried conditions. Exactly 50% of rows include at least one
lower-priority matching user with the opposite polarity; other lower-priority
users independently match or do not match the query. V4 excludes one-user rows:
train examples use 2-4 users and test examples use 2-7 users.

V5 is derived from the V4 splits as a V1-style control. It keeps each V4 row's
priority order, user count, and per-user rule count, but converts
`not_applicable` users into applicable users by replacing one rule per queried
category with the queried value. V5 therefore has no IDK/underspecified users:
all users decide the query with `yes` or `no`.

| Config | Train users | Test users | Conflict-user ratios |
| --- | --- | --- | --- |
| `GeneralAuthorityV3` | 2-5 | 5-50 | 10%, 20%, ..., 90% |
| `ToolAuthorityV3` | 2-5 | 5-50 | 10%, 20%, ..., 90% |

## Prompt Statistics

`Rule types per prompt` counts distinct policy categories in the prompt.
`Rules per prompt` counts individual allow/disallow values; `none` entries are
not counted.

| Config | Split | Rule types per prompt | Rules per prompt | Users |
| --- | --- | --- | --- | --- |
| `GeneralAuthorityV1` | `train` | 2-3, avg 2.2 | 3-51, avg 20.4 | 1-3 |
| `GeneralAuthorityV1` | `test` | 2-4, avg 2.9 | 4-110, avg 44.0 | 1-5 |
| `GeneralAuthorityV2` | `train` | 2-3, avg 2.2 | 3-51, avg 20.2 | 1-3 |
| `GeneralAuthorityV2` | `test` | 2-4, avg 2.9 | 4-110, avg 43.9 | 1-5 |
| `ToolAuthorityV1` | `train` | 2-3, avg 2.2 | 2-54, avg 23.6 | 1-3 |
| `ToolAuthorityV1` | `test` | 2-5, avg 2.8 | 2-110, avg 39.3 | 1-5 |
| `ToolAuthorityV2` | `train` | 2-3, avg 2.2 | 2-54, avg 23.5 | 1-3 |
| `ToolAuthorityV2` | `test` | 2-5, avg 2.8 | 2-110, avg 39.0 | 1-5 |
| `GeneralAuthorityV3` | `train` | 2-3 | sampled, <= 1,000 | 2-5 |
| `GeneralAuthorityV3` | `test` | 2-5 | sampled, <= 1,000 | 5-50 |
| `ToolAuthorityV3` | `train` | 2-3 | sampled, <= 1,000 | 2-5 |
| `ToolAuthorityV3` | `test` | 2-5 | sampled, <= 1,000 | 5-50 |
| `GeneralAuthorityV4` | `train` | 2-3, avg 2.2 | 5-65, avg 26.2 | 2-4 |
| `GeneralAuthorityV4` | `test` | 2-4, avg 2.9 | 6-184, avg 56.1 | 2-7 |
| `GeneralAuthorityV5` | `train` | same as V4 | same as V4 | same as V4 |
| `GeneralAuthorityV5` | `test` | same as V4 | same as V4 | same as V4 |
| `ToolAuthorityV4` | `train` | 2-3, avg 2.2 | 4-92, avg 29.7 | 2-4 |
| `ToolAuthorityV4` | `test` | 2-5, avg 2.8 | 4-181, avg 50.1 | 2-7 |
| `ToolAuthorityV5` | `train` | same as V4 | same as V4 | same as V4 |
| `ToolAuthorityV5` | `test` | same as V4 | same as V4 | same as V4 |

V3 stores the conflict-ratio analysis fields in `meta_data.v3`: `main_user`,
`user_count`, `lower_priority_user_count`, `requested_conflict_ratio`,
`actual_conflict_ratio`, `conflict_user_count`, and
`agreeing_lower_priority_user_count`.

## Columns

| Column | Description |
| --- | --- |
| `text` | Rendered authority setting, priority order, and query. |
| `label` | Gold answer: `Yes` or `No`. |
| `meta_data.query` | Structured query conditions. |
| `meta_data.paraphrase_version` | `v1`, `v2`, `v3`, `v4`, or `v5`. |
| `meta_data.query_meaning_score` | Present for V2; semantic verification score. |
| `meta_data.v3` | Present for V3; main user, user count, and conflict-ratio metadata. |
| `meta_data.v4` | Present for V4; deciding user, priority depth, non-matching higher-priority users, and lower-priority conflict metadata. |
| `meta_data.v5` | Present for V5; V4 source row, user/rule count checks, converted users, and conflict metadata. |

## Load

```python
from datasets import load_dataset

ds = load_dataset("leo-bjpark/authority", "GeneralAuthorityV1")
print(ds["train"][0])
```
