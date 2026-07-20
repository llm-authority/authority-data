# AuthorityBench 데이터 생성 문서

이 문서는 AuthorityBench 데이터셋이 어떤 절차로 생성되는지 정리한다.
핵심 아이디어는 여러 사용자의 allow/disallow 규칙과 사용자 간 우선순위를
함께 제시하고, 주어진 요청이 최종적으로 허용되는지 `Yes`/`No`로 판정하게
하는 것이다.

AuthorityBench는 두 계열의 태스크로 구성된다.

- **GeneralAuthority**: 도구 이름 없이 일반 행동이 허용되는지 판단한다.
  조건 범주는 `day`, `date`, `time`, `month`, `year`이다.
- **ToolAuthority**: 특정 도구 사용이 허용되는지 판단한다. 조건 범주는
  `recipient`, `information_type`, `purpose`, `day`, `time`이고, query에는
  tool name이 함께 들어간다.

각 예시는 정책, 우선순위, query, 정답 label을 포함한다. 모델은 여러 사용자의
규칙이 충돌할 때 우선순위를 따라 최종 결정을 내려야 한다.

## 1. 전체 파이프라인

데이터 생성은 크게 다음 순서로 진행된다.

1. **도메인 정의**
   `src/domains.py`에서 사용할 조건 범주와 각 범주의 가능한 값을 정의한다.
   GeneralAuthority와 ToolAuthority는 서로 다른 category pool을 사용하며,
   ToolAuthority는 train/test에 사용할 tool name도 별도로 정의한다.

2. **category combination 샘플링**
   하나의 예시가 포함할 category set을 고른다. 기본 생성에서는 가능한
   category 조합 중 20개를 샘플링한다. 조합 크기는 2개 이상이며,
   GeneralAuthority와 ToolAuthority 모두 train보다 test가 더 넓은 category
   pool을 갖도록 설계된다.

3. **attribute subset 샘플링**
   선택된 각 category에서 몇 개의 값을 정책에 포함할지 `k`를 샘플링하고,
   실제 attribute value subset을 뽑는다. 예를 들어 `day+time` 조합이면
   `day`에서 몇 개의 요일을, `time`에서 몇 개의 시간대를 포함할지 먼저 정한
   뒤 실제 값을 선택한다.

4. **query value 선택**
   각 category의 샘플된 값 중 하나를 query 값으로 선택한다. Query는
   category마다 하나의 값을 갖는다. ToolAuthority의 경우 여기에 tool name이
   추가된다.

5. **사용자별 polarity 확장**
   user count, 정답 label, conflict 여부를 조합해 여러 정책 케이스를 만든다.
   각 사용자에게 query value에 대한 `yes` 또는 `no` polarity를 배정하고,
   query에 직접 쓰이지 않는 나머지 값의 polarity는 무작위로 채운다.

6. **priority order 샘플링과 label 결정**
   사용자 ID를 무작위 priority order로 섞는다. V1/V2/V3에서는 가장 높은
   우선순위 사용자의 query-level decision이 gold label이 된다. 즉 어떤
   사용자가 query의 모든 조건을 `yes`로 허용하면 그 사용자의 decision은
   `Yes`이고, 하나라도 `no`가 있으면 `No`이다.

7. **rule count 제한과 중복 제거**
   한 예시에 들어가는 전체 rule 수가 제한을 넘으면 제외한다. 이후
   `AuthoritySetting`, `Query`, `Label`, category set, priority, conflict flag를
   기준으로 semantic duplicate을 제거한다.

8. **train/test split**
   split eligibility를 계산한 뒤 train/test에 배치한다. 같은 semantic row가
   train과 test에 동시에 들어가지는 않는다. 다만 category 수, rule 수, user
   count의 쉬운 구간은 train/test에 일부 공존하고, test에는 train에 없는 더 큰
   category 조합, 더 많은 rule, 더 긴 priority chain이 추가된다.

9. **텍스트 렌더링 또는 paraphrase**
   구조화된 base row는 결정적 bullet prompt로 렌더링되어 V1이 된다. V2는 같은
   정책과 label을 유지하되 query 부분만 LLM으로 자연문 paraphrase하고, 의미
   검증을 통과한 row만 유지한다.

10. **추가 stress-test 변형 생성**
    V3와 V4는 V1과 별도의 evaluation stress를 위해 생성된다. V3는 conflict-only
    many-user 설정이고, V4는 우선순위가 높은 사용자가 query에 적용되지 않을 수
    있는 "first applicable user" 설정이다.

## 2. V1 생성 방식

V1은 AuthorityBench의 기본 deterministic split이다. 모든 query는 구조화된
조건 목록으로 렌더링된다.

### 2.1 사용자 정책

각 사용자의 정책은 다음 구조를 갖는다.

```json
{
  "user": "A",
  "authority": "yes",
  "rules": [
    {"category": "day", "value": "monday", "label": "yes"},
    {"category": "time", "value": "13:00-16:59", "label": "no"}
  ]
}
```

`authority`는 해당 사용자가 query에 대해 내리는 decision을 요약한 값이다.
실제 prompt에는 category별 allowed/disallowed value가 bullet 형식으로
렌더링된다.

### 2.2 label 계산

Query를

\[
q=\{(c,x_c): c \in C_q\}
\]

라고 하자. 사용자 \(u\)의 규칙이 query value \(x_c\)에 대해 부여한 label을
\(r_u(c,x_c)\)라고 하면, 사용자 \(u\)의 query-level decision은 다음과 같다.

\[
d_u(q)=
\begin{cases}
\mathsf{Yes}, & \text{if } r_u(c,x_c)=\mathsf{yes}\text{ for every }c\in C_q,\\
\mathsf{No}, & \text{otherwise.}
\end{cases}
\]

Priority order가 \(\pi=(u_1,\ldots,u_m)\)일 때 V1의 gold label은

\[
y=d_{u_1}(q)
\]

이다. Conflict row는 낮은 우선순위 사용자 중 적어도 한 명이 \(u_1\)과 다른
decision을 갖는 경우로 표시한다.

### 2.3 split 설계

V1 split은 완전한 IID random split이 아니라 generalization 축을 조절한 split이다.

| Family | Train categories | Test categories | Train users | Test users |
| --- | --- | --- | ---: | ---: |
| GeneralAuthority | `day`, `date`, `time` | `day`, `date`, `time`, `month`, `year` | 1-3 | 1-5 |
| ToolAuthority | `recipient`, `information_type`, `purpose` | `recipient`, `information_type`, `purpose`, `day`, `time` | 1-3 | 1-5 |

ToolAuthority는 tool name도 split한다.

| Split | Tools |
| --- | --- |
| Train | `send_email`, `send_text_message`, `submit_online_form` |
| Test | `make_phone_call`, `send_voice_message`, `start_live_chat` |

Train eligibility는 작은 설정에 제한된다. 기본값 기준으로 train은 최대 3 users,
train category pool의 subset, train tool pool, 그리고 낮은 rule count를 만족해야
한다. Test는 최대 5 users와 더 넓은 category/tool pool을 허용한다.

## 3. V2: 자연문 query paraphrase

V2는 V1의 policy, priority, label은 그대로 두고 query 표현만 자연문으로 바꾼
버전이다. Paraphrase 모델은 정책, priority, label을 보지 않고 structured query
conditions와 tool name만 입력으로 받는다.

예를 들어 다음 structured query가

```json
{"query_conditions": {"day": "wednesday", "time": "13:00-16:59"}}
```

다음과 같은 자연문 요청으로 바뀔 수 있다.

```text
Action on Wednesday from 13:00 to 16:59.
```

생성된 paraphrase는 별도의 binary semantic verifier로 검사한다. verifier가
원래 structured condition을 모두 보존하고 추가 조건을 넣지 않았다고 판단한
경우만 V2에 유지한다. 그래서 V2 row 수는 V1보다 조금 적다.

## 4. V3: conflict-only many-user stress test

V3는 V1식 deterministic prompt를 유지하지만 모든 row가 conflict case가 되도록
만든다. 주요 목적은 더 긴 priority chain과 다양한 conflict ratio에서 모델이
우선순위 추론을 유지하는지 확인하는 것이다.

V3 생성 방식은 다음과 같다.

- 가장 높은 우선순위 사용자가 gold label을 결정한다.
- 낮은 우선순위 사용자 중 일정 비율은 query에 대해 반대 polarity를 갖는다.
- conflict ratio는 기본적으로 `0.1, 0.2, ..., 0.9`를 사용한다.
- train은 2-5 users, test는 5-50 users 범위를 사용한다.
- prompt가 지나치게 길어지지 않도록 전체 rule 수를 기본 1,000개 이하로 제한한다.

V3 metadata에는 main user, user count, requested/actual conflict ratio,
conflict user count 등이 기록된다.

## 5. V4: first applicable user 설정

V4는 V1 row에서 파생된다. V1/V2/V3에서는 모든 사용자가 query 조건에 대한
decision을 갖는 반면, V4에서는 우선순위가 높은 사용자라도 query에 필요한 모든
조건을 명시적으로 언급하지 않으면 적용되지 않는다.

V4의 결정 규칙은 다음과 같다.

1. priority order를 위에서부터 확인한다.
2. 어떤 사용자의 규칙이 query의 모든 조건을 명시적으로 포함하면 그 사용자가
   applicable user가 된다.
3. 첫 번째 applicable user의 decision이 gold label이다.
4. 더 높은 우선순위 사용자라도 query 조건 일부를 언급하지 않으면 건너뛴다.

생성기는 deciding user를 priority 중 임의 깊이에 배치한다. 그보다 높은
우선순위 사용자들은 query value를 의도적으로 빠뜨린 non-applicable rule set을
받는다. 낮은 우선순위 사용자들은 일부가 query에 match하며, 기본적으로 row의
50%에는 낮은 우선순위의 반대 polarity 사용자가 포함된다.

V4는 one-user row를 만들지 않는다. 기본 user count는 train 2-4 users, test
2-7 users이다.

## 6. row schema

Base JSONL row는 다음 주요 필드를 갖는다.

| Field | Description |
| --- | --- |
| `id` | split 내부 row id |
| `AuthoritySetting` | dataset name, users, rules, optional tool |
| `Query` | structured query attributes와 optional tool |
| `Label` | `Yes` 또는 `No` |
| `metadata.categories` | row에 사용된 category set |
| `metadata.priority` | 높은 우선순위에서 낮은 우선순위 순서의 user list |
| `metadata.is_conflict` | conflict case 여부 |
| `metadata.RuleCount` | prompt에 포함된 전체 allow/disallow rule 수 |
| `metadata.source` | category/sample/case 생성 provenance |
| `metadata.v3` | V3 전용 conflict-ratio metadata |
| `metadata.v4` | V4 전용 applicable-user metadata |

Hugging Face에 push되는 row는 더 가벼운 schema를 사용한다. 현재 `hf_push.py`
기준으로 공통 push schema는 다음과 같고, base metadata의 모든 provenance 필드를
그대로 싣지는 않는다.

| Field | Description |
| --- | --- |
| `text` | 모델에 입력되는 최종 prompt |
| `label` | `Yes` 또는 `No` |
| `meta_data.query` | structured query |
| `meta_data.paraphrase_version` | `v1`, `v2`, `v3`, `v4` |
| `meta_data.query_meaning_score` | V2 semantic verification score |
| `meta_data.v3` | V3 metadata |

V4의 applicable-user provenance는 base JSONL의 `metadata.v4`에 기록된다.
배포용 schema에 V4 분석 필드를 함께 싣고 싶다면 `hf_push.py`의 feature schema도
같이 확장해야 한다.

## 7. 데이터 크기

현재 기본 공개 설정의 row 수는 다음과 같다.

| Config | Train | Test | Total |
| --- | ---: | ---: | ---: |
| GeneralAuthorityV1 | 500 | 1,500 | 2,000 |
| GeneralAuthorityV2 | 491 | 1,468 | 1,959 |
| GeneralAuthorityV3 | 500 | 1,500 | 2,000 |
| GeneralAuthorityV4 | 500 | 1,500 | 2,000 |
| ToolAuthorityV1 | 1,000 | 3,000 | 4,000 |
| ToolAuthorityV2 | 991 | 2,893 | 3,884 |
| ToolAuthorityV3 | 1,000 | 3,000 | 4,000 |
| ToolAuthorityV4 | 1,000 | 3,000 | 4,000 |

V1 deterministic split만 보면 총 6,000 rows이며, GeneralAuthority 2,000 rows와
ToolAuthority 4,000 rows로 구성된다. V2는 semantic verification을 통과한
paraphrase만 유지하므로 총 5,843 rows이다.

## 8. 재생성 명령

구조화 base data는 다음 명령으로 생성한다.

```bash
python authority-data/make_data.py
```

이 명령은 기본적으로 V1 base split을 만들고, V1에서 파생되는 V4와 별도
stress-test인 V3도 함께 생성한다. V3 또는 V4를 생략하려면 각각 `--no-v3`,
`--no-v4`를 사용한다.

LLM query paraphrase cache와 V2 출력은 다음 스크립트에서 만든다.

```bash
python authority-data/paraphrase_data.py --mode generate-and-build
```

Hugging Face 업로드용 dataset dict와 dataset card metadata는 `hf_push.py`에서
구성한다. V2 config를 push하려면 paraphrase cache가 먼저 준비되어 있어야 한다.

## 9. 평가 의도와 한계

AuthorityBench는 다음 능력을 평가하기 위한 controlled synthetic benchmark이다.

- 구조화된 자연어 policy를 읽고 allow/disallow rule을 찾는 능력
- 여러 조건을 조합해 query-level decision을 계산하는 능력
- 사용자 간 conflict를 priority order로 해결하는 능력
- unseen category 조합, unseen tool name, 더 많은 rule, 더 긴 priority chain에
  일반화하는 능력
- V4에서 non-applicable higher-priority user를 건너뛰고 first applicable user를
  찾는 능력

주요 metric은 `Yes`/`No` classification accuracy이다. 함께 보고하면 좋은
breakdown은 dataset config, split, conflict 여부, user count, category
combination, tool name, V3 conflict ratio, V4 deciding-user priority depth이다.
Conflict 여부와 V4 deciding-user depth처럼 base provenance에만 있는 항목으로
분석하려면 base JSONL metadata를 함께 사용해야 한다.

데이터는 synthetic하고 prompt가 템플릿 기반이므로 실제 사용자 지시의 모든
모호성을 포함하지는 않는다. 대신 label을 정확히 통제하고 generalization 축을
분리해 분석할 수 있다는 장점이 있다.
