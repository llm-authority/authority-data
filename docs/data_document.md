# AuthorityBench: Synthetic Data for Priority-Aware Tool Authorization

## Abstract

Language agents increasingly act through external tools, but tool use is rarely
authorized by a single flat rule. Realistic instructions often include multiple
users, conflicting policies, priority ordering, and conditions over time,
recipient, information type, or purpose. We introduce **AuthorityBench**, a
synthetic benchmark for evaluating whether an agent can decide if a requested
action or tool use is authorized under such structured policies. The dataset has
two parts: **GeneralAuthority**, which tests abstract conditional authority over
calendar-like attributes, and **ToolAuthority**, which tests tool-specific
authorization over recipients, information types, purposes, and time conditions.
The train/test splits are designed to stress three forms of generalization:
new category combinations, larger rule sets, and more users. The released split
contains 6,000 examples with balanced labels and conflict rates.

## 1. Motivation

Tool-using agents must often decide not only *what* action is useful, but also
whether the action is permitted. A user may allow an assistant to email schedule
information to a coworker, but not to a family member; another user may override
that rule because they have higher priority. These cases require compositional
reasoning over authority, rather than a simple allowlist of tools.

AuthorityBench targets this setting. Each example presents a policy written as
allow/disallow rules from one or more users, a priority order among those users,
and a query. The model must output whether the requested action is authorized.
When rules conflict, the highest-priority applicable user determines the label.

## 2. Dataset Structure

AuthorityBench contains two datasets. **GeneralAuthority** abstracts away tools
and asks whether an action is authorized under attributes such as day, date, and
time. **ToolAuthority** asks whether using a named tool is authorized under
conditions such as recipient, information type, purpose, day, and time.

Each example contains:

- `text`: the rendered prompt shown to the model;
- `AttributeCombination`: the queried attributes and, for ToolAuthority, the
  tool name;
- `Label`: `Yes` or `No`;
- `metadata`: the structured policy, split information, conflict flag, category
  combination, source row id, and prompt version.

The rendered prompt has three parts: a task statement, user policies, and a
query. For ToolAuthority, the tool appears in the task statement, e.g.,
`Task: decide whether the requested use of the send_text_message tool is
authorized.` The query then lists only the conditions of that tool use.

## 3. Policy Generation

The generator first samples a category combination, then samples attribute
values within the selected categories. For each sampled scenario, it creates
user-specific allow/disallow rules. User priority is sampled independently, and
the label is computed from the highest-priority user whose rule applies to the
query. This produces both agreement cases and conflict cases.

Formally, let \(C \subseteq \mathcal{C}\) be the sampled category set and
\(q=\{(c,x_c):c\in C\}\) be the query attributes. For each user \(u\), the
generator samples a rule set

\[
R_u=\{(c,x,\ell): c\in C,\; \ell\in\{\mathsf{Yes},\mathsf{No}\}\}.
\]

The user's query-level decision is

\[
d_u(q)=
\begin{cases}
\mathsf{Yes}, & \text{if all queried values are allowed by } R_u,\\
\mathsf{No}, & \text{otherwise.}
\end{cases}
\]

Given a sampled priority order \(\pi=(u_1,\ldots,u_m)\), the final label is

\[
y = d_{u_1}(q).
\]

An example is marked as conflicting when two users disagree:

\[
\exists u,v \quad d_u(q)\neq d_v(q).
\]

For ToolAuthority, the same process is conditioned on a sampled tool \(t\),
drawn from the split-specific tool pool. The dataset is not the full Cartesian
product of all possible policies; after candidate generation and deduplication,
we randomly sample the requested number of train and test rows.

## 4. Splits and Generalization Axes

The split design focuses on three axes.

**Category generalization.** GeneralAuthority train uses day, date, and time;
test additionally includes month and year. ToolAuthority train uses recipient,
information type, and purpose; test additionally includes day and time.

**Rule-count generalization.** Test examples include more category combinations
and larger policies, increasing the average number of rules per example.

**User-count generalization.** Train examples contain one to three users, while
test examples contain one to five users. This forces models to handle longer
priority chains at evaluation time.

ToolAuthority also separates tool names across splits. Train uses
`send_email`, `send_text_message`, and `submit_online_form`; test uses
`make_phone_call`, `send_voice_message`, and `start_live_chat`.

## 5. Dataset Size and Balance

The current split contains 6,000 examples. GeneralAuthority has 500 train and
1,500 test examples; ToolAuthority has 1,000 train and 3,000 test examples.
Labels are approximately balanced overall: 2,935 `Yes` examples and 3,065 `No`
examples. Conflict cases are also balanced, with 2,959 conflict examples and
3,041 non-conflict examples.

## 6. Intended Evaluation

AuthorityBench is intended for evaluating whether models can:

1. parse structured natural-language policies;
2. identify applicable allow/disallow rules;
3. resolve disagreement using user priority;
4. generalize to unseen category combinations, unseen tools, more rules, and
   longer priority chains.

The main metric is classification accuracy on `Yes`/`No`. We also recommend
reporting accuracy by dataset, split, conflict status, user count, category
combination, and tool name.

## 7. Limitations

AuthorityBench is synthetic and currently uses templated prompts. This gives
precise control over labels and split structure, but it does not cover all
ambiguities of natural user instructions. The benchmark should therefore be
used as a controlled reasoning test, not as a complete substitute for human
authorization data.

## Appendix A. Dataset Statistics and Examples

The full printed statistics and representative prompts are kept in
[`docs/data_example.md`](data_example.md). The key counts are:

| Dataset | Train | Test | Total |
| --- | ---: | ---: | ---: |
| GeneralAuthority | 500 | 1,500 | 2,000 |
| ToolAuthority | 1,000 | 3,000 | 4,000 |
| **Total** | **1,500** | **4,500** | **6,000** |

Overall label distribution:

| Label | Count | Percent |
| --- | ---: | ---: |
| Yes | 2,935 | 48.9% |
| No | 3,065 | 51.1% |

Overall conflict distribution:

| Conflict | Count | Percent |
| --- | ---: | ---: |
| False | 3,041 | 50.7% |
| True | 2,959 | 49.3% |

Tool split:

| Tool | Train | Test |
| --- | ---: | ---: |
| send_email | 308 | 0 |
| send_text_message | 344 | 0 |
| submit_online_form | 348 | 0 |
| make_phone_call | 0 | 1,017 |
| send_voice_message | 0 | 995 |
| start_live_chat | 0 | 988 |
