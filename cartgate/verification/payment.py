from collections.abc import Mapping
from typing import Protocol

from .models import PaidItem


class PaymentRepository(Protocol):
    def get_paid_items(self, cart_id: str) -> list[PaidItem]: ...


class InMemoryPaymentRepository:
    def __init__(self, carts: Mapping[str, list[PaidItem]] | None = None):
        self._carts = {cart_id: list(items) for cart_id, items in (carts or {}).items()}

    def get_paid_items(self, cart_id: str) -> list[PaidItem]:
        return list(self._carts.get(cart_id, []))

    def set_paid_items(self, cart_id: str, items: list[PaidItem]) -> None:
        self._carts[cart_id] = list(items)
