import numpy as np
import torch

from cdsseg.data.dataset import reverse_complement, tiled_windows
from cdsseg.models import build, one_hot


def test_reverse_complement_is_an_involution():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 5, (3, 50), dtype=np.uint8)
    lab = rng.integers(0, 4, (3, 50), dtype=np.uint8)
    s2, l2 = reverse_complement(*reverse_complement(seq, lab))
    assert (s2 == seq).all() and (l2 == lab).all()


def test_reverse_complement_values():
    seq = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)  # ACGTN
    lab = np.array([[1, 1, 0, 2, 3]], dtype=np.uint8)
    s, l = reverse_complement(seq, lab)
    assert s.tolist() == [[4, 0, 1, 2, 3]]  # N A C G T  (reverse complement of ACGTN)
    assert l.tolist() == [[3, 1, 0, 2, 2]]  # reversed, strand bits swapped


def test_one_hot_masks_n():
    x = one_hot(torch.tensor([[0, 4]], dtype=torch.uint8))
    assert x[0, :, 0].tolist() == [1, 0, 0, 0] and x[0, :, 1].sum() == 0


def test_models_keep_length():
    x = torch.randint(0, 5, (2, 1000), dtype=torch.uint8)
    for name in ["unet", "cnn", "dilated", "bilstm"]:
        assert build(name)(x).shape == (2, 1000)


class _G:
    length = {"chrT": 2300}


def test_tiling_covers_the_end():
    starts = tiled_windows(_G(), "chrT", window=1000, stride=500)
    assert starts[0] == 0 and starts[-1] == 1300
