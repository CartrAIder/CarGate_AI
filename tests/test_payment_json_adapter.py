import pytest

from cartgate.verification import (PaidItem, PaymentJsonError, parse_payment_json,
                                   sku_from_product_id)


def test_parses_payment_json_text():
    payment = parse_payment_json('''{
        "orderId": "order-123",
        "status": "PAID",
        "items": [
            {"id": 31, "productId": 1, "productName": "생수 500ml", "quantity": 2},
            {"id": 32, "productId": 5, "productName": "콜라 500ml", "quantity": 1}
        ]
    }''')
    assert payment.order_id == "order-123"
    assert payment.cart_id == "order-123"
    assert payment.status == "PAID"
    assert payment.items == (PaidItem("S0001", 2), PaidItem("S0005", 1))


def test_product_id_is_converted_to_four_digit_cartgate_sku():
    assert sku_from_product_id(1) == "S0001"
    assert sku_from_product_id(101) == "S0101"


def test_accepts_empty_cart_and_ignores_metadata():
    payment = parse_payment_json({"orderId": "empty-order", "status": "PAID", "items": [],
                                  "paidAt": "2026-08-12"})
    assert payment.items == ()


@pytest.mark.parametrize("payload", [
    "not-json",
    [],
    {"items": []},
    {"orderId": "order", "status": "PENDING_PAYMENT", "items": []},
    {"orderId": "order", "status": "PAID", "items": "not-an-array"},
    {"orderId": "order", "status": "PAID", "items": [{"productId": 1, "productName": "물", "quantity": 0}]},
    {"orderId": "order", "status": "PAID", "items": [{"productId": 1, "productName": "물", "quantity": True}]},
    {"orderId": "order", "status": "PAID", "items": [{"productId": 0, "productName": "물", "quantity": 1}]},
    {"orderId": "order", "status": "PAID", "items": [{"productId": True, "productName": "물", "quantity": 1}]},
    {"orderId": "order", "status": "PAID", "items": [{"productId": 1, "productName": "", "quantity": 1}]},
])
def test_rejects_invalid_payment_payload(payload):
    with pytest.raises(PaymentJsonError):
        parse_payment_json(payload)
