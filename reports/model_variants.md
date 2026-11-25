# Model cost and acceptable-result comparison

All models use the same fixed ten test IDs, shared schema and annotation masks. The strict rule requires valid output, all annotated scalar fields correct, and grouped item F1=1; one test image has no scorable extraction labels, so it cannot be counted as a quality pass. Costs divide spending on **all ten attempts**, including failures/repairs, by strict passes.

| Variant | Model | Attempted | Scalar accuracy | Item F1 | Strict passes/scorable | Inference USD(est) | USD/strict acceptable(est) | End-to-end seconds |
|---|---|---|---|---|---|---|---|---|
| Gemini3.5 | gemini-3.5-flash-lite | 10 | 84.6% | 90.9% | 6/9 | 0.020254 | 0.003376 | 5.591 |
| Gemini3.1 | gemini-3.1-flash-lite | 10 | 73.1% | 72.7% | 5/9 | 0.013695 | 0.002739 | 4.657 |
| NVIDIA8B | nvidia/nemotron-3.5-bolt-e4b | 10 | 61.5% | 27.3% | 2/9 | N/A | N/A | 19.026 |
| OpenRouter free Nano | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | 10 | 65.4% | 0.0% | 2/9 | 0.000000 | 0.000000 | 30.931 |
| Modal self-hosted | Qwen/Qwen3-VL-2B-Instruct | 10 | 73.1% | 90.9% | 4/9 | 0.149956 | 0.037489 | 63.327 |

Gemini rates are paid economic list prices even if account credits apply. OpenRouter free inference is zero within its quota, while OCR hosting still costs money; it is not an unlimited commercial price. NVIDIA commercial rates remain unavailable. Modal includes estimated observed allocated lifecycle, not a verified invoice. All paid-rate estimates exclude human review, hosting of API proxies, and receipt fields without ground-truth coverage.

The OpenRouter key had50 free requests remaining at verification. Both free Gemma4 routes returned429 from their upstream shared pool; the available fixed Nemotron Nano route was tested instead. Extra keys do not increase account-wide daily limits. The free model must earn its place through accuracy and availability. [OpenRouter limits](https://openrouter.ai/docs/api_reference/limits).

Gemini3.1 was evaluated after its development results reached the same15/20 scalar correct count as3.5 when version-only formatting failures were repaired. The repair changed diagnostics only; initial schema/prompt stayed fixed. Its lower paid rates can reduce cost per acceptable result while leaving more receipts for review. Gemini3.5 provides the stronger measured extraction quality. No prompts were tuned on held-out receipts.

Production decision: use the Gemini one-shot architecture when predictable paid cost and limited CPU/RAM are priorities. Choose3.1 for the measured lower inference cost per accepted output, or3.5 for higher measured accuracy. Free NVIDIA/OpenRouter routes remain useful for experiments or low-volume workloads with known quota/availability limits; their local OCR and review costs must also be counted. The default API remainsNVIDIA as requested, with explicit provider selection for Gemini/OpenRouter.

Review effort can outweigh sub-cent token savings. See [total cost and review sensitivity](cost_analysis.md#cost-of-obtaining-acceptable-results);3.5 is the recommended starting model for client quality, while3.1 remains the lower inference-price option.
