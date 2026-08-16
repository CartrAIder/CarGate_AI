from .matcher import verify_items
from .models import VerificationResult, VisionObservation
from .payment import PaymentRepository


class VerificationService:
    def __init__(self, payments: PaymentRepository):
        self._payments = payments

    def verify(self, cart_id: str, observation: VisionObservation) -> VerificationResult:
        return verify_items(self._payments.get_paid_items(cart_id), observation)
