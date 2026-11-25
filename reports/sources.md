# Official sources checked 25 November 2026

Prices are snapshots, not invoices. Token rates below are USD per million tokens for standard online inference. Model availability was also checked against authenticated provider inventories by the main benchmark agent.

| Source | Verified use |
|---|---|
| [Ollama pricing](https://ollama.com/pricing) | Gemma4 input $0.14 / output $0.40, cached $0.05; gpt-oss:20b $0.07 / $0.30, cached $0.035. Pro $20/month includes $60 usage credits; pay-as-you-go has no subscription. |
| [Gemma4 cloud model](https://ollama.com/library/gemma4:cloud) | Official cloud endpoint example uses gemma4; authenticated inventory provided exact gemma4:31b. Avoid local Ollama URLs. |
| [Gemini pricing](https://aistudio.google.com/docs/pricing) | 3.5 Flash-Lite $0.30/$2.50; 3.1 Flash-Lite $0.25/$1.50 for text/image. Thinking tokens use output rates. |
| [Gemini 3.5 model](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite) | Requested exact model identifier exists. |
| [Modal pricing](https://modal.com/pricing) | T4 $0.000164/s; L4 $0.000222/s; physical CPU core $0.0000131/s; RAM $0.00000222/GiB/s. One physical core is two vCPUs. Startup and terminal idle are billable. Region selection surcharge applies. |
| [Modal resources](https://modal.com/docs/guide/resources) | Reservations and limits; CPU/RAM are billed at greater of requested and actual usage. |
| [Modal region selection](https://modal.com/docs/guide/region-selection) | Regional scheduling and price multipliers. |
| [Modal Python API](https://modal.com/docs/reference/modal.Cls) | Model loaded with enter once per container, ASGI API, scale-zero, cache volume, bounded resources. Context7 /websites/modal also checked. |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B, MIT; information extraction prompt supported; advertised eight languages omit Arabic. Dev validation required. |
| [GLM transformer usage](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm_ocr.md) | Native GlmOcrForConditionalGeneration and AutoProcessor. |
| [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | Current root checkpoint 1.5, 1B; old 1.0 subfolder separate. Arabic not explicitly documented. |
| [Hunyuan license](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) | Community terms exclude EU, UK, South Korea; hosted distribution conditions apply. Broad US experiment avoids excluded deployment territory. |
| [Hunyuan inference guide](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/docs/inference/inference.md) | Native Transformers avoids heavy unified CUDA13/vLLM/DFlash runtime. |
| [Qwen3-VL 2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Apache2; fallback only after specialized model dev failures. |
| [NVIDIA CUDA GPU table](https://developer.nvidia.com/cuda/gpus) | T4 compute7.5, L4 compute8.9. |
| [FlashAttention support](https://github.com/Dao-AILab/flash-attention) | Main FA2 kernels require Ampere/Ada/Hopper; T4 experiment uses FP16 eager attention. |
| [DigitalOcean Droplets](https://www.digitalocean.com/pricing/droplets) | Basic Regular 1vCPU/2GiB $12, 2/4 $24, 4/8 $48, 8/16 $96 per month. |
| [Lightsail](https://aws.amazon.com/lightsail/pricing/) | Actual Linux public IPv4 bundles: 2vCPU/2GB $12, 2/4 $24, 2/8 $44, 4/16 $84. Requested ratios are not all sold. |
| [Hetzner regular](https://www.hetzner.com/cloud/regular-performance/) | Official HTML visually-hidden prices include IPv4, exclude VAT: CPX12 1/2 €9.99, CPX22 2/4 €16.99, CPX32 4/8 €30.49, CPX42 8/16 €59.49 for first EU location. Recheck exact region at purchase. |
| [Hetzner cost optimized](https://www.hetzner.com/cloud/cost-optimized/) | CX23 2/4 €5.99, CX33 4/8 €8.99, CX43 8/16 €16.49 including IPv4/excluding VAT; all marked unavailable in official catalogue snapshot. Do not recommend as immediately purchasable. |

No commercial NVIDIA hosted API rate was verified; promotional/developer access must not be interpreted as free production inference. Cloud pricing excludes taxes, backups and unrelated services unless stated. Cloud performance remains estimated unless an actual Linux benchmark is saved.
