# Train/Test Split Design Check

이 문서는 현재 `authority-data/data/base/*/{train,test}.jsonl` 기준으로
train/test가 어떻게 나뉘는지 점검한 내용이다.

## 결론

현재 split은 **동일한 semantic row가 train/test에 동시에 들어가지 않도록**
나뉘어 있다. 다만 다음 세 축이 모두 값 단위로 완전히 disjoint한 것은 아니다.

- scope/category 개수
- 규칙 개수
- 유저 수

정확히는 train을 더 쉬운 영역으로 제한하고, test를 더 넓은 영역까지 포함하게
만든다. 그래서 test에는 train과 겹치는 쉬운 구간도 있고, train에는 없는 더
어려운 extrapolation 구간도 있다.

## Split Logic

생성 로직은 `authority-data/src/make_base_authority_data.py`의
`split_rows()`에 있다.

각 row는 다음 조건으로 train/test eligibility를 계산한다.

### Train eligibility

- `RuleCount <= max_rules_per_scenario // 2`
- `user_count in train_user_counts`
- row의 category set이 train category set의 subset
- ToolAuthority의 경우 tool이 train tool set에 포함

기본 설정에서는:

- `train_user_counts = {1, 2, 3}`
- `max_rules_per_scenario = 110`
- 따라서 train rule limit은 `55`

### Test eligibility

- `RuleCount <= max_rules_per_scenario`
- `user_count in test_user_counts`
- row의 category set이 test category set의 subset
- ToolAuthority의 경우 tool이 test tool set에 포함

기본 설정에서는:

- `test_user_counts = {1, 2, 3, 4, 5}`
- `max_rules_per_scenario = 110`

### Shared-eligible rows

어떤 row가 train/test 양쪽 eligibility를 모두 만족하면 `shared_rows`에 들어간다.
이후 `test_ratio`만큼은 test로, 나머지는 train으로 보낸다.

즉 train/test에 같은 row가 동시에 들어가지는 않지만, 같은 난이도 구간이나 같은
category combination이 양쪽에 존재할 수 있다.

## Scope / Category Split

여기서 scope 개수는 `metadata.categories`의 길이로 보았다.

### Category Pools

GeneralAuthority:

- train categories: `day`, `date`, `time`
- test categories: `day`, `date`, `time`, `month`, `year`

ToolAuthority:

- train categories: `recipient`, `information_type`, `purpose`
- test categories: `recipient`, `information_type`, `purpose`, `day`, `time`

ToolAuthority는 tool name도 split한다.

- train tools: `send_email`, `send_text_message`, `submit_online_form`
- test tools: `make_phone_call`, `send_voice_message`, `start_live_chat`

### Actual Scope Counts

GeneralAuthorityV1:

| Split | Scope Count Distribution |
| --- | --- |
| train | 2: 387, 3: 113 |
| test | 2: 552, 3: 556, 4: 392 |

ToolAuthorityV1:

| Split | Scope Count Distribution |
| --- | --- |
| train | 2: 752, 3: 248 |
| test | 2: 1200, 3: 1230, 4: 439, 5: 131 |

따라서 scope count 자체는 train/test에서 겹친다. 겹치는 값은 `2`, `3`이다.
하지만 test에는 train에 없는 더 큰 scope count가 추가된다.

## Category Combination Overlap

Category combination도 완전히 disjoint하지는 않다.

GeneralAuthorityV1:

- train category combos: 4개
- test category combos: 20개
- overlap: 4개
- overlap combos:
  - `date+time`
  - `day+date`
  - `day+date+time`
  - `day+time`

ToolAuthorityV1:

- train category combos: 4개
- test category combos: 20개
- overlap: 4개
- overlap combos:
  - `information_type+purpose`
  - `recipient+information_type`
  - `recipient+information_type+purpose`
  - `recipient+purpose`

즉 category combination 기준으로도 test는 train-only 영역과 겹치는 쉬운 영역을
일부 포함하고, 동시에 train에는 없는 category를 포함한 조합을 대량으로 포함한다.

## Rule Count Split

Rule count는 `metadata.RuleCount` 기준이다.

기본 생성 설정에서 `max_rules_per_user = 22`, test 최대 user 수가 5명이므로:

- `max_rules_per_scenario = 22 * 5 = 110`
- train rule limit = `110 // 2 = 55`
- test rule limit = `110`

Actual ranges:

| Dataset | Train Rule Count Range | Test Rule Count Range |
| --- | ---: | ---: |
| GeneralAuthorityV1 | 3-51 | 4-110 |
| ToolAuthorityV1 | 2-54 | 2-110 |

따라서 rule count 값도 train/test에서 완전히 disjoint하지 않다. 다만 train은
55 이하로 제한되고, test는 110까지 허용되어 더 큰 rule set을 포함한다.

## User Count Split

기본 설정:

- train: 1-3 users
- test: 1-5 users

Actual distributions:

GeneralAuthorityV1:

| Split | User Count Distribution |
| --- | --- |
| train | 1: 92, 2: 214, 3: 194 |
| test | 1: 150, 2: 300, 3: 327, 4: 368, 5: 355 |

ToolAuthorityV1:

| Split | User Count Distribution |
| --- | --- |
| train | 1: 181, 2: 424, 3: 395 |
| test | 1: 326, 2: 669, 3: 692, 4: 665, 5: 648 |

따라서 user count도 `1`, `2`, `3`은 train/test 양쪽에 있고, test에만 `4`, `5`
user 케이스가 추가된다.

## Row-Level Overlap Check

동일 semantic row overlap은 없다.

Semantic row key는 다음 필드를 기준으로 확인했다.

- `AuthoritySetting`
- `Query`
- `Label`
- `metadata.categories`
- `metadata.priority`
- `metadata.is_conflict`

결과:

| Dataset | Semantic Row Overlap |
| --- | ---: |
| GeneralAuthorityV1 | 0 |
| ToolAuthorityV1 | 0 |

## Interpretation

현재 split은 다음과 같이 해석하는 것이 정확하다.

1. **Exact row leakage는 막는다.**
   같은 policy/query/label row가 train과 test에 동시에 들어가지는 않는다.

2. **쉬운 구간은 train/test에 모두 있다.**
   scope count 2-3, user count 1-3, 낮은 rule count는 test에도 존재한다.

3. **test에는 extrapolation 구간이 추가된다.**
   train에 없는 더 많은 categories, 더 많은 rules, 더 긴 user priority chain이
   test에 들어간다.

따라서 문장으로 요약하면:

> Train/test는 row-level로는 disjoint하지만, scope count/rule count/user count
> 값 자체를 완전히 disjoint하게 나눈 것은 아니다. Train은 작은 설정으로 제한하고,
> test는 그 설정을 포함하면서 더 큰 scope, rule set, user count로 확장한다.
