"""Validate payment JSON and convert it to verification domain models."""
import json
from dataclasses import dataclass
from typing import Any

from .models import PaidItem


class PaymentJsonError(ValueError):
    """Raised when a payment payload does not match the expected contract."""


@dataclass(frozen=True)
class ParsedPayment:
    order_id: str
    status: str
    items: tuple[PaidItem, ...]

    @property
    def cart_id(self) -> str:
        """Verification uses the backend order ID as its cart identifier."""
        return self.order_id


def sku_from_product_id(product_id: int) -> str:
    """Convert QuickPass's product PK to CartGate's S-prefixed SKU."""
    if isinstance(product_id, bool) or not isinstance(product_id, int) or product_id < 1:
        raise PaymentJsonError("productId must be a positive integer")
    return f"S{product_id:04d}"


def parse_payment_json(payload: str | bytes | bytearray | dict[str, Any]) -> ParsedPayment:
    """Parse a paid QuickPass order response into verification domain models.

    Unknown fields are ignored so the payment API can add metadata without
    breaking verification. Required fields and their value types are strict.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PaymentJsonError(f"invalid payment JSON: {exc}") from exc
    elif isinstance(payload, dict):
        data = payload
    else:
        raise PaymentJsonError("payment payload must be JSON text, bytes, or an object")

    if not isinstance(data, dict):
        raise PaymentJsonError("payment JSON root must be an object")

    order_id = data.get("orderId")
    if not isinstance(order_id, str) or not order_id.strip():
        raise PaymentJsonError("orderId must be a non-empty string")

    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        raise PaymentJsonError("status must be a non-empty string")
    status = status.strip()
    if status != "PAID":
        raise PaymentJsonError(f"order is not paid: status={status}")

    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise PaymentJsonError("items must be an array")

    items = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise PaymentJsonError(f"items[{index}] must be an object")
        product_id = raw_item.get("productId")
        product_name = raw_item.get("productName")
        quantity = raw_item.get("quantity")
        try:
            sku = sku_from_product_id(product_id)
        except PaymentJsonError as exc:
            raise PaymentJsonError(f"items[{index}].{exc}") from exc
        if not isinstance(product_name, str) or not product_name.strip():
            raise PaymentJsonError(f"items[{index}].productName must be a non-empty string")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise PaymentJsonError(f"items[{index}].quantity must be a positive integer")
        items.append(PaidItem(sku, quantity))

    return ParsedPayment(order_id.strip(), status, tuple(items))


def paid_items_from_json(payload: str | bytes | bytearray | dict[str, Any]) -> list[PaidItem]:
    """Convenience adapter when the caller only needs the paid item list."""
    return list(parse_payment_json(payload).items)
