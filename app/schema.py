"""Versioned common extraction contract. Monetary values serialize as strings."""
from decimal import Decimal, InvalidOperation
import re
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def parse_decimal(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
        value = value.replace("٬", "").replace("٫", ".").replace("\u200f", "").replace("\u200e", "").strip()
        value = re.sub(r"\s+", "", value)
        if "," in value and "." not in value:
            parts = value.split(",")
            value = "".join(parts) if all(len(p) == 3 for p in parts[1:]) else value.replace(",", ".")
        else:
            value = value.replace(",", "")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("Money must be finite")
    return result


Money = Annotated[Decimal | None, BeforeValidator(parse_decimal)]


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_description: str | None = None
    quantity: Money = None
    unit_price: Money = None
    line_total: Money = None


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    merchant_name: str | None = None
    receipt_number: str | None = None
    purchase_date: str | None = Field(default=None, description="ISO YYYY-MM-DD only if unambiguous; otherwise preserve printed date")
    purchase_time: str | None = None
    currency: str | None = Field(default=None, description="Only explicitly visible currency; never infer from country")
    subtotal: Money = None
    discount: Money = None
    tax: Money = None
    total: Money = None
    line_items: list[LineItem] = Field(default_factory=list)
    raw_text: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


def provider_schema() -> dict:
    """JSON schema generated in application, never supplied by API consumers."""
    return Receipt.model_json_schema(mode="serialization")
