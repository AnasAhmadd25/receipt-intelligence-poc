from benchmark.cost import token_cost, project_cost, Budget
import pytest

def test_thinking_is_not_double_billed():
    meta={'input_tokens':1000000,'output_tokens':1000000,'thinking_tokens':500000,'usage_complete':True}
    assert token_cost(meta,.3,2.5)==2.8

def test_unknown_usage_not_zero_cost():
    assert token_cost({'input_tokens':0,'output_tokens':0,'usage_complete':False},.3,2.5) is None

def test_cumulative_budget_reservations(tmp_path):
    b=Budget(tmp_path/'budget.json',limit=5)
    b.reserve('gpu',2.5)
    b.reserve('api',1.5)
    with pytest.raises(ValueError):b.reserve('extra',1.1)
    b.settle('api',.01)
    b.reserve('extra',1)

def test_unknown_projection_stays_unknown():
    assert project_cost(None,1000) is None
