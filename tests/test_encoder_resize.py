"""Replay observed canonical TorchCPU encoder preprocessing without Torch at runtime."""
from pathlib import Path
import json
import mlx.core as mx
import numpy as np
import pytest
from moge_mlx.model import _pil_resize

FIXTURE = Path(__file__).parent / 'fixtures' / 'encoder_resize_torch.npz'
CASES = json.loads(FIXTURE.with_suffix('.json').read_text())['cases']

@pytest.mark.parametrize('case', CASES, ids=[c['name'] for c in CASES])
def test_encoder_resize_matches_canonical_antialias(case):
    with np.load(FIXTURE) as data:
        x = data[case['name'] + '_input']
        expected = data[case['name'] + '_expected']
    actual = np.array(_pil_resize(mx.array(x), *case['target']))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=5e-7)
