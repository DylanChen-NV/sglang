import unittest

from sglang.srt.models.kimi_k3 import _use_shared_experts_attn_tp_comm


class TestKimiK3SharedExpertSp(unittest.TestCase):
    def test_a2a_sp_uses_attn_tp_collectives(self):
        self.assertTrue(
            _use_shared_experts_attn_tp_comm(
                enabled=True,
                ep_a2a=True,
                attn_tp_size=32,
            )
        )

    def test_collectives_require_flag_a2a_and_multiple_ranks(self):
        for enabled, ep_a2a, attn_tp_size in (
            (False, True, 32),
            (True, False, 32),
            (True, True, 1),
        ):
            with self.subTest(
                enabled=enabled,
                ep_a2a=ep_a2a,
                attn_tp_size=attn_tp_size,
            ):
                self.assertFalse(
                    _use_shared_experts_attn_tp_comm(
                        enabled=enabled,
                        ep_a2a=ep_a2a,
                        attn_tp_size=attn_tp_size,
                    )
                )


if __name__ == "__main__":
    unittest.main()
