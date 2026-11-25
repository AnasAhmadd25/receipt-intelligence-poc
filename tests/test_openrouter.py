import httpx
import pytest
from app.providers.openrouter import OpenRouterProvider

@pytest.mark.asyncio
async def test_fixed_free_model_has_no_paid_fallback(monkeypatch):
    seen=[]
    async def post(client,url,**kwargs):
        seen.append(kwargs['json'])
        return httpx.Response(200,json={'model':'google/gemma-4-26b-a4b-it:free','choices':[{'message':{'content':'{"total":"2.50"}'}}],'usage':{'prompt_tokens':10,'completion_tokens':9,'cost':0}},request=httpx.Request('POST',url))
    monkeypatch.setenv('OPENROUTER_API_KEY','test')
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    r=await OpenRouterProvider().extract(b'data','image/jpeg',ocr={'text':'Total2.50','lines':[]})
    assert r['receipt']['total']=='2.50'
    assert seen[0]['model'].endswith(':free')
    assert 'models' not in seen[0]
    assert r['metadata']['reported_cost_usd']==0
