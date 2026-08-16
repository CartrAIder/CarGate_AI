import pytest

from cartgate.verification import (Decision, DetectedProduct, InMemoryPaymentRepository,
                                   PaidItem, VerificationService, VisionObservation)


@pytest.mark.parametrize("paid,vision,expected", [
    ([PaidItem("water", 1), PaidItem("coke", 1)], [DetectedProduct("water", 1), DetectedProduct("coke", 1)], Decision.PASS),
    ([PaidItem("water", 1)], [DetectedProduct("water", 1), DetectedProduct("coke", 1)], Decision.REVIEW),
    ([PaidItem("water", 1)], [DetectedProduct("water", 2)], Decision.REVIEW),
    ([PaidItem("water", 2)], [DetectedProduct("water", 1)], Decision.PASS),
    ([PaidItem("water", 1)], [DetectedProduct("water", 1), DetectedProduct(None, 1)], Decision.REVIEW),
    ([], [DetectedProduct("coke", 1)], Decision.REVIEW),
    ([], [], Decision.PASS),
])
def test_required_scenarios(paid, vision, expected):
    service = VerificationService(InMemoryPaymentRepository({"cart": paid}))
    assert service.verify("cart", VisionObservation(tuple(vision))).decision is expected


def test_invalid_vision_fails_safe():
    service = VerificationService(InMemoryPaymentRepository())
    assert service.verify("cart", VisionObservation.invalid("camera unavailable")).decision is Decision.REVIEW


def test_input_order_does_not_change_result():
    service = VerificationService(InMemoryPaymentRepository({"cart": [PaidItem("water", 1)]}))
    products = [DetectedProduct("coke", 1), DetectedProduct("water", 1)]
    assert service.verify("cart", VisionObservation(tuple(products))) == service.verify(
        "cart", VisionObservation(tuple(reversed(products))))
