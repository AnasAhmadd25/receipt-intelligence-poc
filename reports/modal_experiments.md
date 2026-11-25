# Modal self-hosted experiments

These are development-only candidate results from fixed balanced Arabic/Latin samples. They are separate from the ten held-out comparison. Every monetary value is an observed-resource list-price estimate, not an invoice.

| Model | Attempted | Generated | Valid schema | Correct/scorable scalar fields | Mean infer seconds | Peak RSS MiB incl init | Peak allocated GPU MiB | Observed lifecycle USD(est) |
|---|---|---|---|---|---|---|---|---|
| GLM-OCR | 4 | 4 | 4 | 1/6 | 46.879 | 8,407.258 | 2,621.741 | 0.05298 |
| HunyuanOCR | 4 | 3 | 0 | 0/4 | 17.554 | 8,215.652 | 2,301.206 | 0.05271 |
| Qwen3-VL-2B-Instruct | 4 | 4 | 4 | 3/6 | 45.969 | 11,129.668 | 4,505.738 | 0.08806 |
| Qwen3-1.7B | 4 | 4 | 3 | 0/6 | 29.416 | 11,003.109 | 5,036.156 | 0.04633 |

GLM-OCR initially exceeded T4 memory with eager attention on a high-resolution Arabic image. Switching toSDPA and a1megapixel cap made inference run on the T4; low development extraction accuracy then excluded it. Hunyuan returned OCR-box arrays instead of the receipt schema on its three retained generated outputs; its first cold request returned a303deferred response before the driver was corrected to follow redirects. That lost response is availability/transport evidence, not an extraction-quality result.

Qwen3-VL-2B was selected after the specialized models failed the schema or accuracy checks. Its four development receipts produced valid receipt JSON and3/6correct annotated scalars. Original images and resized dimensions are recorded. The service uses FP16 andSDPA because T4 compute capability7.5does not support the intended BF16/FlashAttention2 path. First-ever model download/init and cached cold starts differ; metadata separates initialization, warm inference and client round-trip.

A separate fallback experiment uses the chosen PaddleOCR CPU engine with locally hostedQwen3-1.7B onT4. It has12GiB allocated RAM, while the direct VLM uses8GiB. It calls no paid external inference APIs. The image-to-text and text-generation costs are inside its container lifecycle, and original OCR text is preserved. Its known language configuration is Arabic; that recognizer also contains Latin characters. The fallback is an independently measured alternative, not silently substituted into the direct ten-receipt benchmark.

Both deployments have max_containers=1,min_containers=0,2second scale-down, persistent weights in a ModalVolume, pinned HuggingFace revisions, and bearer authentication. Unauthenticated401and authenticated200were verified for the direct service. No credentials appear in source or reports. The API client handles Modal deferred-response redirects, and retains invalid model output/resource evidence.

Provisioned resources differ from observed utilization. Observed GPU allocation uses PyTorch peaks; sampled utilization comes fromnvidia-smi. CPU includes process/children core-seconds. RAM samples are process-tree RSS. Lifecycle estimates group stable container IDs and use their maximum observed uptime plus terminal idle, avoiding repeated charging of initialization per warm receipt. Startup before Python and unobserved failed-container work are not complete billing records. The experiment cap has conservative allowances for them in `results/spend_ledger.json`.

Scale-to-zero reduces idle costs for batch arrivals; sporadic requests with long gaps can repeatedly pay model initialization. CPU-only VLM inference andL4speedups were not measured, so they remain N/A rather than being recommended from specifications. Separate x86Linux Paddle andTesseract concurrency1/2/4probes are fully retained and support CPU OCR sizing.

Sources: [Modal pricing](https://modal.com/pricing), [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR), [Qwen VLM](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [fallback text model](https://huggingface.co/Qwen/Qwen3-1.7B).
