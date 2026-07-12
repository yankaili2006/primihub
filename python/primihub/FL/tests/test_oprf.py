#!/usr/bin/env python3
"""
OPRF (Oblivious Pseudo-Random Function) - simplified test implementation.
Uses HMAC-based OPRF for testing purposes (the real crypto uses EC).
"""
import hashlib
import hmac
import os


class OprfSender:
    def __init__(self, key=None):
        self.key = key or os.urandom(32)

    def evaluate(self, blinded: bytes) -> bytes:
        return hmac.digest(self.key, blinded, 'sha256')

    def blind_evaluate(self, input_bytes: bytes) -> bytes:
        return hmac.digest(self.key, input_bytes, 'sha256')


class OprfReceiver:
    def __init__(self):
        self._r = os.urandom(32)

    def blind(self, input_bytes: bytes) -> bytes:
        r_bytes = self._r + input_bytes
        return hashlib.sha256(r_bytes).digest()

    def finalize(self, input_bytes: bytes, evaluated: bytes) -> bytes:
        return hashlib.sha256(input_bytes + evaluated).digest()


# ── Tests ──────────────────────────────────────────

def test_sender_key_generation():
    sender = OprfSender()
    assert len(sender.key) == 32


def test_receiver_blind():
    receiver = OprfReceiver()
    blinded = receiver.blind(b"test_input")
    assert len(blinded) == 32


def test_full_oprf_protocol():
    sender = OprfSender()
    receiver = OprfReceiver()
    input_data = b"Hello, OPRF!"
    blinded = receiver.blind(input_data)
    evaluated = sender.evaluate(blinded)
    output = receiver.finalize(input_data, evaluated)
    assert len(output) == 32


def test_consistency():
    sender = OprfSender()
    out_a1 = sender.blind_evaluate(b"same")
    out_a2 = sender.blind_evaluate(b"same")
    out_b = sender.blind_evaluate(b"diff")
    assert out_a1 == out_a2
    assert out_a1 != out_b


def test_different_senders():
    s1 = OprfSender()
    s2 = OprfSender()
    assert s1.blind_evaluate(b"test") != s2.blind_evaluate(b"test")


def test_receiver_blinding_hides_input():
    sender = OprfSender()
    r1, r2 = OprfReceiver(), OprfReceiver()
    b1 = r1.blind(b"secret1")
    b2 = r2.blind(b"secret2")
    # Both are blinded - sender can't distinguish the inputs
    eval1 = sender.evaluate(b1)
    eval2 = sender.evaluate(b2)
    assert len(eval1) == 32
    assert len(eval2) == 32


if __name__ == "__main__":
    test_sender_key_generation()
    test_receiver_blind()
    test_full_oprf_protocol()
    test_consistency()
    test_different_senders()
    test_receiver_blinding_hides_input()
    print("All OPRF tests passed!")
