import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import phe
    from primihub.FL.crypto.paillier import Paillier
    HAS_PAILLIER = True
except (ImportError, ModuleNotFoundError):
    HAS_PAILLIER = False

try:
    from primihub.FL.utils.base import BaseModel
    HAS_BASE = True
except (ImportError, ModuleNotFoundError):
    HAS_BASE = False


@pytest.mark.skipif(not HAS_PAILLIER, reason="phe library not installed")
class TestPaillier:
    def setup_method(self):
        pub, priv = phe.paillier.generate_paillier_keypair(n_length=256)
        self.p = Paillier(pub, priv)

    def test_encrypt_decrypt_scalar(self):
        plain = 12345
        cipher = self.p.encrypt_scalar(plain)
        decrypted = self.p.decrypt_scalar(cipher)
        assert decrypted == plain

    def test_encrypt_decrypt_vector(self):
        vec = [1.0, 2.5, -3.2, 100.0]
        cipher_vec = self.p.encrypt_vector(vec)
        decrypted = self.p.decrypt_vector(cipher_vec)
        for orig, dec in zip(vec, decrypted):
            assert abs(orig - dec) < 1e-6

    def test_encrypt_decrypt_matrix(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        cipher_mat = self.p.encrypt_matrix(mat)
        decrypted = self.p.decrypt_matrix(cipher_mat)
        np.testing.assert_array_almost_equal(mat, decrypted, decimal=6)

    def test_homomorphic_add_scalar(self):
        a, b = 10.0, 20.0
        ca = self.p.encrypt_scalar(a)
        cb = self.p.encrypt_scalar(b)
        c_sum = ca + cb
        decrypted = self.p.decrypt_scalar(c_sum)
        assert abs(decrypted - (a + b)) < 1e-6

    def test_homomorphic_mul_const(self):
        val = 5.0
        cipher = self.p.encrypt_scalar(val)
        result = cipher * 3.0
        decrypted = self.p.decrypt_scalar(result)
        assert abs(decrypted - 15.0) < 1e-6


@pytest.mark.skipif(not HAS_BASE, reason="base module not importable")
class TestBaseModel:
    def test_base_model_concrete(self):
        class ConcreteModel(BaseModel):
            def run(self):
                pass
        kwargs = {
            'roles': {'client': ['node1']},
            'common_params': {},
            'role_params': {},
            'node_info': {},
            'task_info': None,
        }
        model = ConcreteModel(**kwargs)
        assert model.roles == {'client': ['node1']}

    def test_base_model_requires_run(self):
        with pytest.raises(TypeError):
            BaseModel(
                roles={},
                common_params={},
                role_params={},
                node_info={},
                task_info=None,
            )
