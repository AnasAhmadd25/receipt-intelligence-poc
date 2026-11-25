# Cost analysis

All monetary results below are **estimated economic charges**, not account billing records. Expenditure cannot be confirmed without provider invoices/balance records. Measurements include usage from repairs/retries where returned; failures with missing usage remain unknown. Price check: 2026-11-25.

| Provider | Volume | Inference USD (estimated) |
|---|---|---|
| ocr_ollama | 50 | N/A |
| ocr_ollama | 100 | N/A |
| ocr_ollama | 1000 | N/A |
| ocr_ollama | 10000 | N/A |
| ocr_ollama | 100000 | N/A |
| ocr_nvidia | 50 | N/A |
| ocr_nvidia | 100 | N/A |
| ocr_nvidia | 1000 | N/A |
| ocr_nvidia | 10000 | N/A |
| ocr_nvidia | 100000 | N/A |
| gemini | 50 | 0.10127 |
| gemini | 100 | 0.20254 |
| gemini | 1000 | 2.02543 |
| gemini | 10000 | 20.25430 |
| gemini | 100000 | 202.54300 |
| modal_hybrid | 50 | 0.74978 |
| modal_hybrid | 100 | 1.49956 |
| modal_hybrid | 1000 | 14.99564 |
| modal_hybrid | 10000 | 149.95643 |
| modal_hybrid | 100000 | 1,499.56435 |

Ollama gemma4:31b uses $0.14/M input + $0.40/M output; gpt-oss:20b $0.07/$0.30. Pay-as-you-go is distinct from Pro $20/month with $60 credits and Max $100 with $300 credits. Subscription cash and included inference credits must not be double-counted. [Official Ollama prices](https://ollama.com/pricing).

Gemini 3.5 Flash-Lite uses $0.30/M image/text input + $2.50/M output including thoughts; 3.1 uses $0.25/$1.50. Thinking is already included once in output_tokens. Image/prompt modality token detail is retained in attempt metadata. [Official Google prices](https://aistudio.google.com/docs/pricing).

NVIDIA developer access succeeded; commercial token pricing and cash debits are unavailable. Promotional/credit access is not evidence that production inference is free. NVIDIA projections remain N/A.

Local OCR has no metered invoice on this Mac; its CPU work still has economic cost. At Modal CPU list rates the variable compute estimate is `core_seconds×0.0000131 + allocated_GiB×allocated_seconds×0.00000222`, plus any region multiplier. A continuously running server instead costs its monthly plan, allocated over actual monthly volume. For the $48/month 4vCPU/8GiB example, infrastructure adds $48, $4.80 or $0.48 per 1,000 at 1k,10k or100k receipts/month respectively, before API inference. These are different utilization scenarios, not additional simultaneous bills.

## Modal arrival scenarios

| Scenario (estimated) | Allocated seconds | USD |
|---|---|---|
| 50 continuous, one container | 2,969.241 | 0.71011 |
| 50 sporadic, each container has gone idle | 3,597.419 | 0.86034 |
| 50 sustained, warm serial service | 2,956.421 | 0.70704 |

The sporadic case assumes inter-arrival gaps exceed the configured 2s scale-down period; not every request is assumed to cause a cold start in the batch or sustained cases. T4 is $0.000164/s, L4 $0.000222/s, CPU physical core $0.0000131/s, RAM GiB $0.00000222/s. US placement has a1.15× example multiplier; scenarios use recorded allocations and region multipliers, and remain unavailable without stable container IDs. Model load occurs once per container, cache persists in a volume, and min_containers=0. Model revisions/weights/init cost and observed container uptimes are retained. Actual billed lifecycle may include unobserved startup; reported estimates are lower bounds where applicable. Credits reduce cash expenditure but not resource cost. [Official Modal prices](https://modal.com/pricing).

GPU L4 is 35.4% more expensive per GPU-second than T4; it needs measured latency savings to justify that change. It was not selected solely from specifications. CPU-only document VLM is not claimed practical without measurement.

At measured warm latency the self-host variable cost is $0.014141/receipt vs Gemini $0.002025. **There is no positive volume break-even under these measured rates**: increasing volume does not overcome a higher variable cost. Accuracy and operational effort also matter.

The cumulative authorization cap and reserved unknown-price costs are in `results/spend_ledger.json`. Development runs are separate from the selected evaluation-run cost; their raw usage is retained in `results/development_*` and Modal experiment files.

<!-- variant-cost-analysis -->
## Cost of obtaining acceptable results

On the measured ten-input mix,3.1 costs $1.370/1,000 attempts and3.5 costs $2.025. The extra3.5 inference fee is only $0.656/1,000 attempts. It also produced one additional verified strict pass. These are small-sample observations, not stable production error rates.

An illustrative review-cost calculation makes3.5 cheaper overall once correcting/reviewing an additional unverified receipt costs more than **$0.0066 (0.66 cents)**. Formula: model cost + non-verified fraction × review cost. An unscorable receipt is conservatively unverified; no real wage or review duration is assumed.

| Assumed review/correction USD per unverified result | 3.1 USD/1,000 attempts incl review(est) | 3.5 USD/1,000 attempts incl review(est) |
|---|---|---|
| 0 | 1.370 | 2.025 |
| 0.01 | 6.370 | 6.025 |
| 0.05 | 26.370 | 22.025 |
| 0.10 | 51.370 | 42.025 |

This sensitivity supports **Gemini3.5 as the stronger client recommendation** when correctness and review effort matter.3.1 is the cheapest measured paid inference per verified acceptable output, with lower accuracy. Neither schema validity nor absence of arithmetic warnings proves correctness in production.

| Monthly raw attempts | 3.1 inference+$12 proxy USD(est) | 3.1 USD/1k strict accepted(est) | 3.5 inference+$12 proxy USD(est) | 3.5 USD/1k strict accepted(est) |
|---|---|---|---|---|
| 1000 | 13.370 | 26.739 | 14.025 | 23.376 |
| 10000 | 25.695 | 5.139 | 32.254 | 5.376 |
| 100000 | 148.950 | 2.979 | 214.543 | 3.576 |

The fixed proxy is the1vCPU/2GB DigitalOcean example; serverless CPU hosting can avoid that fixed fee at low volume. Review is excluded from this hosting table. Capacity, quota and rejection-rate extrapolations require larger client data. A free OCR/text route still needs OCR compute: the measured Linux sizing suggests2vCPU/4GB at$24/month (DO example), and the50/day OpenRouter allowance restricts monthly capacity. The tested self-hosted VLM had higher variable cost and fewer strict passes than3.5, so no positive volume break-even appears at these measured settings.

## Experiment expenditure evidence

Across recorded hosted trials, reported token usage corresponds to$0.100829at paid list rates. Four server-error attempts omitted usage and remain unknown. Retained Modal lifecycles and conservative allowances are separate. The cumulative tracker reserved/estimated$2.2904034against the authorized$5cap; this is a conservative accounting bound, not an invoice or a claim that the entire reservation was spent. See `results/expenditure_summary.json` and `results/spend_ledger.json`.
