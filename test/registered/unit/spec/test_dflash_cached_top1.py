"""Cached raw model predictions must be invalidated by sampling transforms."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    can_use_q35rt_fused_cycle,
    get_dflash_greedy_target_top1,
)


class SamplingInfo(SimpleNamespace):
    def __len__(self):
        return 1


class TestDflashCachedTop1(unittest.TestCase):
    def test_fused_cycle_sampling_guards(self):
        base = dict(is_all_greedy=True, has_custom_logit_processor=False,
                    acc_linear_penalties=None, penalizer_orchestrator=None,
                    grammar_mask=None, logit_bias=None)
        self.assertTrue(can_use_q35rt_fused_cycle(SamplingInfo(**base), has_grammar=False))
        for name, value in (
            ('is_all_greedy', False), ('has_custom_logit_processor', True),
            ('acc_linear_penalties', object()), ('grammar_mask', object()),
            ('logit_bias', object()), ('penalizer_orchestrator', SimpleNamespace(is_required=True)),
        ):
            with self.subTest(name=name):
                self.assertFalse(can_use_q35rt_fused_cycle(
                    SamplingInfo(**dict(base, **{name: value})), has_grammar=False))
        self.assertFalse(can_use_q35rt_fused_cycle(SamplingInfo(**base), has_grammar=True))
        self.assertFalse(can_use_q35rt_fused_cycle(
            SamplingInfo(**base), has_grammar=False, return_sampling_mask=True))

    def run_case(self, *, adjustment=None, grammar=False):
        logits = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
        raw = logits.argmax(-1)
        info = SamplingInfo(has_custom_logit_processor=False,
                            acc_linear_penalties=None, penalizer_orchestrator=None,
                            grammar_mask=None, logit_bias=None)
        if adjustment == 'bias':
            info.logit_bias = torch.tensor([[5., 0., 0.]])
        elif adjustment == 'accumulated':
            info.acc_linear_penalties = torch.tensor([[0., 6., 0.]])
        elif adjustment == 'penalizer':
            info.penalizer_orchestrator = SimpleNamespace(is_required=True)
            info.apply_logits_bias = lambda value: value.add_(torch.tensor([[0., 7., 0.]]))
        elif adjustment == 'custom':
            info.has_custom_logit_processor = True
        with patch('sglang.srt.speculative.dflash_utils.apply_custom_logit_processor',
                   side_effect=lambda value, *_args, **_kwargs: value[:, 1].add_(8.)):
            modified = apply_dflash_verify_logits_adjustments(
                next_token_logits=logits, sampling_info=info, draft_token_num=2)
        if grammar:
            logits[:, [0, 2]] = -float('inf')
        output = SimpleNamespace(next_token_logits=logits,
                                 customized_info={'q35rt_target_top1': raw})
        actual = get_dflash_greedy_target_top1(output, logits_modified=modified,
                                             grammar_mask=object() if grammar else None)
        self.assertTrue(torch.equal(actual, logits.argmax(-1)))
        if adjustment is None and not grammar:
            self.assertIs(actual, raw)
        else:
            self.assertFalse(torch.equal(raw, actual))

    def test_raw_reuse(self):
        self.run_case()

    def test_adjustments_invalidate_raw_prediction(self):
        for adjustment in ('bias', 'accumulated', 'penalizer', 'custom'):
            with self.subTest(adjustment=adjustment):
                self.run_case(adjustment=adjustment)

    def test_grammar_invalidate_raw_prediction(self):
        self.run_case(grammar=True)


if __name__ == '__main__':
    unittest.main()
