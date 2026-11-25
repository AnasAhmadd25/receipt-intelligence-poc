"""Compare cheaper models using identical test IDs and explicit acceptance rules."""
import csv
import json
from pathlib import Path
from benchmark.report import ROOT, read, summarize, selected_records, fmt, table
from benchmark.acceptance import summarize_acceptance


def main():
    manifest=read(ROOT/'data/manifest.json')
    experiments=[('test10','gemini','Gemini3.5'),('test10_gemini31','gemini','Gemini3.1'),('test10','ocr_nvidia','NVIDIA8B'),('test10','ocr_openrouter','OpenRouter free Nano'),('test10','modal_hybrid','Modal self-hosted')]
    rows=[]
    csvrows=[]
    for run,provider,label in experiments:
        path=ROOT/'results'/run
        if not path.exists():continue
        expected,split=selected_records(path,manifest)
        s=summarize(path,provider,expected)
        a=summarize_acceptance(s['rows'],s.get('cost'))
        rows.append([label,s.get('model','N/A'),str(s['attempted']),fmt(s.get('field_accuracy'),True),fmt(s.get('line_f1'),True),f"{a['strict_passes']}/{a['strict_scorable']}",fmt(s.get('cost'),digits=6),fmt(a['cost_per_strict_accepted_usd'],digits=6),fmt(s.get('mean_latency'))])
        csvrows.append({'run':run,'provider':provider,'model':s.get('model'),'attempted':s['attempted'],'scorable':a['strict_scorable'],'strict_passes':a['strict_passes'],'field_accuracy':s.get('field_accuracy'),'line_f1':s.get('line_f1'),'inference_estimated_usd':s.get('cost'),'cost_per_accepted_usd':a['cost_per_strict_accepted_usd'],'mean_latency_seconds':s.get('mean_latency')})
    text='# Model cost and acceptable-result comparison\n\nAll models use the same fixed ten test IDs, shared schema and annotation masks. The strict rule requires valid output, all annotated scalar fields correct, and grouped item F1=1; one test image has no scorable extraction labels, so it cannot be counted as a quality pass. Costs divide spending on **all ten attempts**, including failures/repairs, by strict passes.\n\n'
    text+=table(['Variant','Model','Attempted','Scalar accuracy','Item F1','Strict passes/scorable','Inference USD(est)','USD/strict acceptable(est)','End-to-end seconds'],rows)
    text+='\nGemini rates are paid economic list prices even if account credits apply. OpenRouter free inference is zero within its quota, while OCR hosting still costs money; it is not an unlimited commercial price. NVIDIA commercial rates remain unavailable. Modal includes estimated observed allocated lifecycle, not a verified invoice. All paid-rate estimates exclude human review, hosting of API proxies, and receipt fields without ground-truth coverage.\n\nThe OpenRouter key had50 free requests remaining at verification. Both free Gemma4 routes returned429 from their upstream shared pool; the available fixed Nemotron Nano route was tested instead. Extra keys do not increase account-wide daily limits. The free model must earn its place through accuracy and availability. [OpenRouter limits](https://openrouter.ai/docs/api_reference/limits).\n\nGemini3.1 was evaluated after its development results reached the same15/20 scalar correct count as3.5 when version-only formatting failures were repaired. The repair changed diagnostics only; initial schema/prompt stayed fixed. Its lower paid rates can reduce cost per acceptable result while leaving more receipts for review. Gemini3.5 provides the stronger measured extraction quality. No prompts were tuned on held-out receipts.\n\nProduction decision: use the Gemini one-shot architecture when predictable paid cost and limited CPU/RAM are priorities. Choose3.1 for the measured lower inference cost per accepted output, or3.5 for higher measured accuracy. Free NVIDIA/OpenRouter routes remain useful for experiments or low-volume workloads with known quota/availability limits; their local OCR and review costs must also be counted. The default API remainsNVIDIA as requested, with explicit provider selection for Gemini/OpenRouter.\n'
    (ROOT/'reports/model_variants.md').write_text(text)
    with (ROOT/'results/model_variants.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    projections=ROOT/'results/cost_projections.csv'
    cheaper=next((r for r in csvrows if r.get('model')=='gemini-3.1-flash-lite' and r.get('inference_estimated_usd') is not None),None)
    if projections.exists() and cheaper:
        existing=list(csv.DictReader(projections.open()));columns=list(existing[0]);existing=[r for r in existing if r['provider']!='gemini31']
        for n in [50,100,1000,10000,100000]:
            existing.append({'provider':'gemini31','volume':n,'inference_cost_usd_estimated':cheaper['inference_estimated_usd']/cheaper['attempted']*n,'evaluated_receipts':cheaper['attempted'],'basis':'Measured Gemini3.1 usage x paid list rates; estimated extrapolation excluding hosting/review'})
        with projections.open('w') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(existing)
    paid={r['model']:r for r in csvrows if r.get('model') in ['gemini-3.1-flash-lite','gemini-3.5-flash-lite']}
    if len(paid)==2:
        low,high=paid['gemini-3.1-flash-lite'],paid['gemini-3.5-flash-lite']
        lo,hi=low['inference_estimated_usd']/low['attempted'],high['inference_estimated_usd']/high['attempted']
        lp,hp=low['strict_passes']/low['attempted'],high['strict_passes']/high['attempted']
        section='\n<!-- variant-cost-analysis -->\n## Cost of obtaining acceptable results\n\n'
        section+=f'On the measured ten-input mix,3.1 costs ${lo*1000:.3f}/1,000 attempts and3.5 costs ${hi*1000:.3f}. The extra3.5 inference fee is only ${(hi-lo)*1000:.3f}/1,000 attempts. It also produced one additional verified strict pass. These are small-sample observations, not stable production error rates.\n\n'
        if hp>lp:
            threshold=(hi-lo)/(hp-lp)
            section+=f'An illustrative review-cost calculation makes3.5 cheaper overall once correcting/reviewing an additional unverified receipt costs more than **${threshold:.4f} ({threshold*100:.2f} cents)**. Formula: model cost + non-verified fraction × review cost. An unscorable receipt is conservatively unverified; no real wage or review duration is assumed.\n\n'
        review=[]
        for c in [0,.01,.05,.10]:
            review.append([fmt(c,digits=2),fmt((lo+(1-lp)*c)*1000,digits=3),fmt((hi+(1-hp)*c)*1000,digits=3)])
        section+=table(['Assumed review/correction USD per unverified result','3.1 USD/1,000 attempts incl review(est)','3.5 USD/1,000 attempts incl review(est)'],review)
        section+='\nThis sensitivity supports **Gemini3.5 as the stronger client recommendation** when correctness and review effort matter.3.1 is the cheapest measured paid inference per verified acceptable output, with lower accuracy. Neither schema validity nor absence of arithmetic warnings proves correctness in production.\n\n'
        hosting=[]
        for n in [1000,10000,100000]:
            hosting.append([n,fmt(12+n*lo),fmt((12+n*lo)/(n*lp)*1000),fmt(12+n*hi),fmt((12+n*hi)/(n*hp)*1000)])
        section+=table(['Monthly raw attempts','3.1 inference+$12 proxy USD(est)','3.1 USD/1k strict accepted(est)','3.5 inference+$12 proxy USD(est)','3.5 USD/1k strict accepted(est)'],hosting)
        section+='\nThe fixed proxy is the1vCPU/2GB DigitalOcean example; serverless CPU hosting can avoid that fixed fee at low volume. Review is excluded from this hosting table. Capacity, quota and rejection-rate extrapolations require larger client data. A free OCR/text route still needs OCR compute: the measured Linux sizing suggests2vCPU/4GB at$24/month (DO example), and the50/day OpenRouter allowance restricts monthly capacity. The tested self-hosted VLM had higher variable cost and fewer strict passes than3.5, so no positive volume break-even appears at these measured settings.\n'
        cost_path=ROOT/'reports/cost_analysis.md'
        if cost_path.exists():
            old=cost_path.read_text().split('\n<!-- variant-cost-analysis -->')[0]
            cost_path.write_text(old+section)
        with (ROOT/'reports/model_variants.md').open('a') as f:
            f.write('\nReview effort can outweigh sub-cent token savings. See [total cost and review sensitivity](cost_analysis.md#cost-of-obtaining-acceptable-results);3.5 is the recommended starting model for client quality, while3.1 remains the lower inference-price option.\n')
    print('Generated model_variants.md and model_variants.csv')

if __name__=='__main__':main()
