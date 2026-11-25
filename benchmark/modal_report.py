"""Summarize retained self-hosted development evidence without invented labels."""
import json
from benchmark.report import ROOT,read,table,fmt,has_model_output,modal_resource_cost
from benchmark.metrics import evaluate


def main():
    dev={r['id']:r for r in read(ROOT/'data/manifest.json')['development']}
    models=['GLM-OCR','HunyuanOCR','Qwen3-VL-2B-Instruct','Qwen3-1.7B']
    rows=[]
    for model in models:
        results=read(ROOT/f'results/modal_dev_{model}.json',[])
        if not results:continue
        metrics=[];generated=[]
        for r in results:
            if has_model_output(r):
                generated.append(r)
                metrics.append({'sample_id':r['sample_id'],'metrics':evaluate(r.get('receipt') or {},read(ROOT/dev[r['sample_id']]['annotation']))})
        (ROOT/f'results/modal_dev_{model}_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
        c=sum(r['metrics']['field_correct'] for r in metrics);n=sum(r['metrics']['field_total'] for r in metrics)
        lat=[r['metadata']['latency_seconds'] for r in generated if r.get('metadata',{}).get('latency_seconds') is not None]
        resources=[r.get('metadata',{}).get('resource',{}) for r in generated]
        peaks=[r.get('metadata',{}).get('initialization_resource',{}).get('peak_ram_bytes',0) for r in generated]+[q.get('peak_ram_bytes',0) for q in resources]
        rows.append([model,len(results),len(generated),sum(bool(r.get('metadata',{}).get('json_valid')) for r in generated),f'{c}/{n}',fmt(sum(lat)/len(lat) if lat else None),fmt(max(peaks,default=0)/2**20),fmt(max((q.get('gpu_peak_allocated_bytes',0) or 0 for q in resources),default=0)/2**20),fmt(modal_resource_cost(generated),digits=5)])
    text='# Modal self-hosted experiments\n\nThese are development-only candidate results from fixed balanced Arabic/Latin samples. They are separate from the ten held-out comparison. Every monetary value is an observed-resource list-price estimate, not an invoice.\n\n'
    text+=table(['Model','Attempted','Generated','Valid schema','Correct/scorable scalar fields','Mean infer seconds','Peak RSS MiB incl init','Peak allocated GPU MiB','Observed lifecycle USD(est)'],rows)
    text+='\nGLM-OCR initially exceeded T4 memory with eager attention on a high-resolution Arabic image. Switching toSDPA and a1megapixel cap made inference run on the T4; low development extraction accuracy then excluded it. Hunyuan returned OCR-box arrays instead of the receipt schema on its three retained generated outputs; its first cold request returned a303deferred response before the driver was corrected to follow redirects. That lost response is availability/transport evidence, not an extraction-quality result.\n\nQwen3-VL-2B was selected after the specialized models failed the schema or accuracy checks. Its four development receipts produced valid receipt JSON and3/6correct annotated scalars. Original images and resized dimensions are recorded. The service uses FP16 andSDPA because T4 compute capability7.5does not support the intended BF16/FlashAttention2 path. First-ever model download/init and cached cold starts differ; metadata separates initialization, warm inference and client round-trip.\n\nA separate fallback experiment uses the chosen PaddleOCR CPU engine with locally hostedQwen3-1.7B onT4. It has12GiB allocated RAM, while the direct VLM uses8GiB. It calls no paid external inference APIs. The image-to-text and text-generation costs are inside its container lifecycle, and original OCR text is preserved. Its known language configuration is Arabic; that recognizer also contains Latin characters. The fallback is an independently measured alternative, not silently substituted into the direct ten-receipt benchmark.\n\nBoth deployments have max_containers=1,min_containers=0,2second scale-down, persistent weights in a ModalVolume, pinned HuggingFace revisions, and bearer authentication. Unauthenticated401and authenticated200were verified for the direct service. No credentials appear in source or reports. The API client handles Modal deferred-response redirects, and retains invalid model output/resource evidence.\n\nProvisioned resources differ from observed utilization. Observed GPU allocation uses PyTorch peaks; sampled utilization comes fromnvidia-smi. CPU includes process/children core-seconds. RAM samples are process-tree RSS. Lifecycle estimates group stable container IDs and use their maximum observed uptime plus terminal idle, avoiding repeated charging of initialization per warm receipt. Startup before Python and unobserved failed-container work are not complete billing records. The experiment cap has conservative allowances for them in `results/spend_ledger.json`.\n\nScale-to-zero reduces idle costs for batch arrivals; sporadic requests with long gaps can repeatedly pay model initialization. CPU-only VLM inference andL4speedups were not measured, so they remain N/A rather than being recommended from specifications. Separate x86Linux Paddle andTesseract concurrency1/2/4probes are fully retained and support CPU OCR sizing.\n\nSources: [Modal pricing](https://modal.com/pricing), [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR), [Qwen VLM](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [fallback text model](https://huggingface.co/Qwen/Qwen3-1.7B).\n'
    (ROOT/'reports/modal_experiments.md').write_text(text)
    print('Generated Modal development evidence report')

if __name__=='__main__':main()
