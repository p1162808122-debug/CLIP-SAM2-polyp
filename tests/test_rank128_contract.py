import sys

import MyTest
import MyTrain
from models.model import PolypSAM2Model
from tools import profile_batch_size


def test_rank128_is_accepted_by_all_polyp3_entrypoints(monkeypatch):
    assert 128 in PolypSAM2Model.SUPPORTED_LORA_RANKS

    monkeypatch.setattr(sys, "argv", ["MyTrain.py", "--lora-rank", "128"])
    assert MyTrain.parse_args().lora_rank == 128

    monkeypatch.setattr(sys, "argv", ["MyTest.py", "--lora-rank", "128"])
    assert MyTest.parse_args().lora_rank == 128

    monkeypatch.setattr(sys, "argv", ["profile_batch_size.py", "--rank", "128"])
    assert profile_batch_size.parse_args().rank == 128
