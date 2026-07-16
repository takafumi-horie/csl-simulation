from __future__ import annotations

from legacy_csl.cli import build_parser, main


def test_parser_supports_all_public_commands():
    parser = build_parser()

    assert parser.prog == "csl"
    for command in ("learn", "predict", "analyze", "validate-data"):
        args = parser.parse_args([command, "--config", "example.yaml"])
        assert args.command == command


def test_make_fixture_command(tmp_path):
    destination = tmp_path / "fixture"

    assert main(["make-fixture", str(destination), "--documents", "5"]) == 0

    assert (destination / "agentU.npz").is_file()
    assert (destination / "agentA.npz").is_file()
    assert (destination / "agentB.npz").is_file()
    assert (destination / "single_words.json").is_file()
