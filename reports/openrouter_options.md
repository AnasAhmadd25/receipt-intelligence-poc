# OpenRouter: free and inexpensive candidates

Checked 2026-11-25. Public catalog research was followed by authenticated trials after the user supplied a key locally. The free Gemma4 routes were upstream-rate-limited; the available fixed Nemotron Nano Omni route was evaluated on8development and10held-out receipts. See the executed update and model comparison for measured results. Credential values are excluded.

## Candidates actually listed

| Candidate | Current identifier | Input/output USD per million tokens | Relevant scope |
|---|---|---|---|
| Gemma 4 A4B free | `google/gemma-4-26b-a4b-it:free` | $0 / $0 | 25.2B total, 3.8B active; image or OCR-text candidate |
| Nemotron Nano Omni free | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | $0 / $0 | 30B total, approximately 3B active; image or OCR-text candidate |
| Gemma 4 31B free | `google/gemma-4-31b-it:free` | $0 / $0 | Larger model; comparison candidate, not a small dense model |
| Gemma 3 4B paid | `google/gemma-3-4b-it` | $0.04 / $0.05 on the listed Parasail route | Small dense model; text + image input; structured output advertised |
| Llama 3.1 8B paid | `meta-llama/llama-3.1-8b-instruct` | $0.05 / $0.08 in the catalog snapshot; model page also lists DeepInfra at $0.02 / $0.04 | Text-only; route must be fixed to make price assumptions reproducible |

The live [public catalog](https://openrouter.ai/api/v1/models) contained those identifiers; listing does not establish authenticated account access or receipt accuracy. Model descriptions and endpoint details are available on the official [Gemma A4B free page](https://openrouter.ai/google/gemma-4-26b-a4b-it:free), [Nemotron free page](https://openrouter.ai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free), [Gemma 3 4B page](https://openrouter.ai/google/gemma-3-4b-it), and [Llama 3.1 8B page](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct). Active-parameter counts do not represent total model memory requirements.

The obvious `qwen/qwen3-4b` candidate is **currently unavailable**: its official page says no provider has served it since July 6, 2026. Do not assume an older recommended model is still accessible, and do not manufacture a free variant by appending a suffix to an unlisted identifier. [Official Qwen 3 4B availability](https://openrouter.ai/qwen/qwen3-4b), [free-variant rules](https://openrouter.ai/docs/guides/routing/model-variants/free).

## Free allowance and reproducibility

Published free-model limits are 20 requests/minute and 50 requests/day; the daily allowance rises to 1,000 once the account has purchased at least $10 of credits. Failed attempts also consume daily allowance. These are account-level limits; extra keys do not multiply capacity. Query the account's authenticated `GET /api/v1/key` counter after a key is available rather than assuming its remaining quota. A later authenticated request verified an initial allowance of50remaining free requests; see results/openrouter_discovery.json. [Official rate-limit reference](https://openrouter.ai/docs/api_reference/limits), [official pricing FAQ](https://openrouter.ai/pricing).

A development set of 8 plus a held-out set of 10 needs 18 initial requests, up to 36 with one repair each, before other account activity or availability failures. A 50-image run with repairs can exceed the basic daily allowance. Purchasing $10 solely to raise quota would exceed this experiment's original $5 cap and needs separate authorization; an already-eligible account is a different case.

Use a fixed free model identifier for experiments. The `openrouter/free` router selects among available free models, so it does not isolate model quality. Paid entries in a fallback list can incur charges even when the first entry is free; a free-only experiment must keep the entire fallback list free. Free availability can differ from a paid twin. [Official free-variant behavior](https://openrouter.ai/docs/guides/routing/model-variants/free), [Free Models Router](https://openrouter.ai/openrouter/free).

OpenRouter bills posted model-token rates; its Standard fee is charged when buying credits, currently 5.5%, rather than being added individually to every request. Paid routing can select different hosts and rates, so pin the provider or record the actual returned cost. Do not mix token inference price, purchased-credit cash and free quota. [Official pricing and fees](https://openrouter.ai/pricing).

## Price sensitivity using existing tokens, not invented accuracy

The existing NVIDIA held-out run saved 10 receipts with **36,857 input tokens and 8,024 output tokens**, including recorded repairs. Every saved row reported complete usage. Applying alternative prices to those counts produces this counterfactual token-cost sensitivity:

| Assumed paid route | Hypothetical total for those token counts | Hypothetical cost per input | Hypothetical token cost per 1,000 inputs |
|---|---|---|---|
| Gemma 3 4B, $0.04/$0.05 per M | $0.00187548 | $0.000187548 | $0.187548 |
| Llama 3.1 8B, catalog $0.05/$0.08 per M | $0.00248477 | $0.000248477 | $0.248477 |
| Llama 3.1 8B, listed DeepInfra $0.02/$0.04 per M | $0.00105810 | $0.000105810 | $0.105810 |

Formula: `(36,857 × input_rate + 8,024 × output_rate) / 1,000,000`. Rates come from the official [Gemma 3 4B endpoint table](https://openrouter.ai/google/gemma-3-4b-it), [Llama 3.1 8B endpoint table](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct), and the [catalog snapshot](https://openrouter.ai/api/v1/models). These are price scenarios, **not actual OpenRouter expenditure or measured OpenRouter outputs**. Different tokenizers, prompts, output lengths, repairs and model quality can change every result. OCR compute, monthly infrastructure and credit-purchase fees are excluded.

The subsequently tested free Nano route produced2/9strict passes with zero trial token charges, but poor line-item accuracy. No paid Gemma/Llama accuracy or commercial quality-adjusted cost was measured. NVIDIA's acceptance rate must not be transferred to those models. See [acceptance definitions and measured-provider results](acceptance.md) for the strict rule and the weaker total-only sensitivity.

## Initial candidate plan and unmeasured paid alternatives

The most useful free trial is fixed Gemma A4B on the same development receipts, comparing direct image input with identical cached OCR text. The most useful paid comparison is a fixed cheap Gemma 3 4B route with recorded actual usage. These are candidates, not winners: choose only after measuring the strict acceptance rate and full operating cost. A free token price does not eliminate OCR infrastructure, reliability limits or review costs. Keep the already-tested Gemini paid baseline for a calculable cost per accepted receipt, and keep NVIDIA's experimental free debit separate from unknown commercial pricing.

## Executed update

A key was later supplied through a hidden local Terminal prompt. Authenticated inspection reported50free requests remaining. Both fixed freeGemma4 routes rejected inference withupstream429shared-pool limits. The fixed freeNemotron NanoOmni route succeeded on a tiny probe and was evaluated on8development plus10held-out receipts. See `reports/model_variants.md`, `results/development_openrouter_nano/`, `results/test10/ocr_openrouter/` and `results/openrouter_route_probes.json` for measured results. Earlier proposal sections above are dated pre-inference research, not a claim that the current experiment remains unrun. Multiple keys were not rotated.
