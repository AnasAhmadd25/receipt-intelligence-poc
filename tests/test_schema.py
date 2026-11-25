from decimal import Decimal
import pytest
from app.schema import Receipt, parse_decimal

def test_arabic_digits_decimal_safe():
    assert parse_decimal('١٬٢٣٤٫٥٠') == Decimal('1234.50')
    assert parse_decimal('12,500') == Decimal('12500')
    assert parse_decimal('12,50') == Decimal('12.50')
    assert parse_decimal('') is None

def test_rejects_non_finite_money():
    with pytest.raises(ValueError):
        Receipt(total='NaN')

def test_nullable_and_original_text_preserved():
    r = Receipt(raw_text='السعر ١٢٫٥٠', total='١٢٫٥٠')
    assert r.total == Decimal('12.50')
    assert r.merchant_name is None
    assert r.raw_text == 'السعر ١٢٫٥٠'
    assert r.model_dump(mode='json')['total'] == '12.50'
