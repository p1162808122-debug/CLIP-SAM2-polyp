import inspect
import sys


def test_polyp4_resolves_independent_rank_defaults_and_legacy_rank():
    from models.polyp4 import polyp4_rank_label, resolve_polyp4_ranks

    assert resolve_polyp4_ranks() == (128, 128, 64)
    assert resolve_polyp4_ranks(legacy_lora_rank=128) == (128, 128, 128)
    assert resolve_polyp4_ranks(sam_lora_rank=512, clip_lora_rank=256, text_lora_rank=64) == (512, 256, 64)
    assert polyp4_rank_label(512, 256, 64) == "sam512_clip256_text64"


def test_polyp4_constructor_exposes_three_independent_rank_arguments():
    from models.polyp4 import Polyp4Model

    parameters = inspect.signature(Polyp4Model.__init__).parameters
    assert "sam_lora_rank" in parameters
    assert "clip_lora_rank" in parameters
    assert "text_lora_rank" in parameters


def test_train_and_test_entrypoints_parse_independent_ranks(monkeypatch):
    import MyTest
    import MyTrain

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "MyTrain.py",
            "--sam-lora-rank",
            "512",
            "--clip-lora-rank",
            "256",
            "--text-lora-rank",
            "64",
        ],
    )
    train_args = MyTrain.parse_args()
    assert (train_args.sam_lora_rank, train_args.clip_lora_rank, train_args.text_lora_rank) == (512, 256, 64)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "MyTest.py",
            "--sam-lora-rank",
            "512",
            "--clip-lora-rank",
            "256",
            "--text-lora-rank",
            "64",
        ],
    )
    test_args = MyTest.parse_args()
    assert (test_args.sam_lora_rank, test_args.clip_lora_rank, test_args.text_lora_rank) == (512, 256, 64)


def test_profile_entrypoint_parses_three_ranks(monkeypatch):
    from tools import profile_batch_size

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_batch_size.py",
            "--sam-lora-rank",
            "512",
            "--clip-lora-rank",
            "256",
            "--text-lora-rank",
            "64",
            "--batch-size",
            "8",
            "--worker",
        ],
    )
    args = profile_batch_size.parse_args()
    assert (args.sam_lora_rank, args.clip_lora_rank, args.text_lora_rank) == (512, 256, 64)
