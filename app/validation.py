from decimal import Decimal
from app.schema import Receipt


def validate_receipt(receipt: Receipt) -> Receipt:
    warnings = list(receipt.validation_warnings)
    for index, item in enumerate(receipt.line_items):
        if item.quantity is not None and item.unit_price is not None and item.line_total is not None:
            if abs(item.quantity * item.unit_price - item.line_total) > Decimal("0.02"):
                warnings.append(f"line_items[{index}]: quantity × unit_price differs from line_total")
    if all(getattr(receipt, k) is not None for k in ("subtotal", "tax", "discount", "total")):
        if abs(receipt.subtotal + receipt.tax - receipt.discount - receipt.total) > Decimal("0.02"):
            warnings.append("subtotal + tax - discount differs from total")
    receipt.validation_warnings = list(dict.fromkeys(warnings))
    scalars = ["merchant_name", "receipt_number", "purchase_date", "purchase_time", "currency", "subtotal", "discount", "tax", "total"]
    receipt.missing_fields = list(dict.fromkeys(receipt.missing_fields + [k for k in scalars if getattr(receipt, k) is None]))
    return receipt
