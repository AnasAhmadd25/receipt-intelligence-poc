# Infrastructure recommendation

Sizing below uses measured OCR core-seconds and peak process-tree memory with 25% RAM headroom and a 75% CPU target. **Cloud performance values are estimates, not direct Apple-to-x86 equivalence.** Linux Modal CPU validation is retained separately. Listed monthly prices were checked 2026-11-25, are region-specific, and exclude taxes/network extras.

Sizing anchor: **measured x86 Linux Modal, concurrency 1, one physical core/two vCPU equivalent**. Selected Paddle development evidence: 1912.3MiB peak RAM; 7.389 core-seconds/receipt; minimum planned memory with 25% headroom 2.33GiB. The Mac peaked near 5GiB, while Linux was lower, demonstrating why direct Apple-to-x86 sizing was inappropriate. The adapter serializes OCR inference under a lock; multiple waiting HTTP requests do not imply multiple model workers.

| Cloud | vCPU/GiB | Monthly | Safe active OCR workers (est) | OCR rpm (est ceiling) | Daily at70% availability (est) | CPU% machine(est) | RAM headroom MiB (est) | Infra/1k at 10k/mo | GPU? |
|---|---|---|---|---|---|---|---|---|---|
| DigitalOcean | 1/2 | USD 12 | 0 | N/A | N/A | N/A | N/A | USD 1.200 | No |
| DigitalOcean | 2/4 | USD 24 | 1 | 6.090 | 6,139 | 75.000 | 2,183.742 | USD 2.400 | No |
| DigitalOcean | 4/8 | USD 48 | 1 | 6.090 | 6,139 | 37.500 | 6,279.742 | USD 4.800 | No |
| DigitalOcean | 8/16 | USD 96 | 1 | 6.090 | 6,139 | 18.750 | 14,471.742 | USD 9.600 | No |
| Hetzner | 1/2 | EUR 9.99 | 0 | N/A | N/A | N/A | N/A | EUR 0.999 | No |
| Hetzner | 2/4 | EUR 16.99 | 1 | 6.090 | 6,139 | 75.000 | 2,183.742 | EUR 1.699 | No |
| Hetzner | 4/8 | EUR 30.49 | 1 | 6.090 | 6,139 | 37.500 | 6,279.742 | EUR 3.049 | No |
| Hetzner | 8/16 | EUR 59.49 | 1 | 6.090 | 6,139 | 18.750 | 14,471.742 | EUR 5.949 | No |
| Hetzner | 2/4 | EUR 5.99 | unavailable | 6.090 | 6,139 | 75.000 | 2,183.742 | EUR 0.599 | No |
| Hetzner | 4/8 | EUR 8.99 | unavailable | 6.090 | 6,139 | 37.500 | 6,279.742 | EUR 0.899 | No |
| Hetzner | 8/16 | EUR 16.49 | unavailable | 6.090 | 6,139 | 18.750 | 14,471.742 | EUR 1.649 | No |
| AWS Lightsail | 2/2 | USD 12 | 0 | N/A | N/A | N/A | N/A | USD 1.200 | No |
| AWS Lightsail | 2/4 | USD 24 | 1 | 6.090 | 6,139 | 75.000 | 2,183.742 | USD 2.400 | No |
| AWS Lightsail | 2/8 | USD 44 | 1 | 6.090 | 6,139 | 75.000 | 6,279.742 | USD 4.400 | No |
| AWS Lightsail | 4/16 | USD 84 | 1 | 6.090 | 6,139 | 37.500 | 14,471.742 | USD 8.400 | No |
| AWS Lightsail | 8/32 | USD 164 | 1 | 6.090 | 6,139 | 18.750 | 30,855.742 | USD 16.400 | No |

All rpm values are conservative CPU-only estimates based on the Linux core-seconds anchor where available, capped at one OCR worker and 75% utilization of its physical-core-equivalent capacity. Cloud vendor vCPU sharing varies; validate throughput on the purchased plan. External API latency/rate limits lower end-to-end throughput. Daily capacity assumes 70% availability for retries/queue gaps; it is not an SLA. Memory headroom is provisioned RAM minus measured peak.

## Deployment choices

- **Low volume:** Gemini-only API, no dedicated GPU and no local OCR dependencies. Start 1vCPU/2GB ($12/month DigitalOcean example) or a scale-to-zero CPU wrapper. Actual client concurrency 1/2/4 measurements are retained; remote inference CPU/RAM are managed by Google. The local client CPU profile includes SDK startup and API waiting.
- **Medium volume requiring local OCR:** **2vCPU/4GB, one active OCR worker**, bounded upstream requests, using the measured Linux1-worker memory anchor. DigitalOcean Basic $24/month or Hetzner CPX22 €16.99/month are examples. At1vCPU/2GB the 25% RAM margin is insufficient for this Paddle corpus. Concurrent preprocessing raises RAM even though the model lock serializes inference; cap active OCR to 1 on 4GB. 4vCPU/8GB ($48/month DO / €30.49 Hetzner) gives room for concurrent preprocessing and API tasks, but the present one-worker lock gains no OCR throughput from extra CPUs.
- **High volume:** load-test API quotas and shard OCR only after validating per-worker memory.8vCPU/16GB is not justified by this one-worker POC. Multiple process workers are a further experiment, not a claimed result. For owning inference, use the measured scale-zero T4 configuration only if its accuracy and warm unit cost justify it; L4 and CPU-only VLM need separate evidence before recommendation.

Lightsail's actual bundles differ from requested CPU/RAM ratios; for example 8GB is 2vCPU, and 16GB is 4vCPU. Hetzner CX plans shown as unavailable are not purchase recommendations. Currency conversion is not guessed; EUR and USD remain separate.

## Linux validation and limits

| Platform (Tesseract) | Concurrency | Provisioned vCPU | Provisioned MiB | rpm measured | Peak MiB measured | Core-sec/receipt |
|---|---|---|---|---|---|---|
| Linux-4.19.0-gvisor-x86_64-with-glibc2.36 | 1 | 2 | 4096 | 26.629 | 360.059 | 2.240 |
| Linux-4.19.0-gvisor-x86_64-with-glibc2.36 | 2 | 2 | 4096 | 26.420 | 566.469 | 3.985 |
| Linux-4.19.0-gvisor-x86_64-with-glibc2.36 | 4 | 2 | 4096 | 19.628 | 840.902 | 3.420 |

Selected Paddle on Linux:

| Concurrency | Provisioned vCPU | Provisioned MiB | rpm measured | Peak MiB measured | Core-sec/receipt |
|---|---|---|---|---|---|
| 1 | 2 | 8192 | 7.663 | 1,912.258 | 7.389 |
| 2 | 2 | 8192 | 8.564 | 2,812.023 | 6.918 |
| 4 | 2 | 8192 | 8.549 | 3,169.840 | 6.969 |

Both Linux probes used the same eight development images. Their provisioned RAM was not their observed consumption; this supports a smaller deployment recommendation but does not replace testing a 4GiB allocation directly. The remote model evidence is in [Modal experiments](modal_experiments.md). Modal physical CPU core represents two vCPUs; local percent100 means one fully occupied core.

See [cost analysis](cost_analysis.md) for continuous/sporadic/sustained GPU scenarios and break-even. See [official sources](sources.md) for plan scope and prices.

For the1vCPU/2GiB Gemini proxy, a conservative planning estimate is concurrency2,25receipts/minute and25,200receipts/day at70%availability, subject to APIquota. This is anchored to the measured client trial, not an x86 server benchmark. Estimated CPU utilization is20–50%of one vCPU, with over1.5GiB memory headroom against the measured216MiB client peak plus web overhead. At10,000rawreceipts/month, the$12proxy adds$1.20/1,000 before paid inference; no dedicatedGPU is needed.

The2vCPU/4GiB OCR recommendation uses `MAX_IN_FLIGHT=1` for memory safety. Including measured NVIDIA APIwaiting, its planning throughput is approximately3receipts/minute (about3,000/day at70%availability), below the OCR-only ceiling in the table; CPU utilization should remain below saturation. These are estimates requiring target-host validation. Use8GiB when allowing multiple queued/preprocessed large images.

The direct VLM experiment peaked near7.6GiB processRSS. A production trial needs roughly12GiB RAM to provide25%headroom, plusT4(16GiBVRAM; measuredpeakabout4.5GiB) and the experimentally allocated2physicalCPUcores. This is a conditional machine specification, not a recommendation to self-host: measured quality-adjusted cost favored the external API. GPUconcurrency2/4, CPU-only VLM andL4performance were not measured; the present service serializes one model instance, so no GPUconcurrency speedup is claimed.
