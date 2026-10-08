from pathlib import Path

import yaml

from evalsuite.request_schemas import GPT_VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def test_gpts_default_variant_matches_its_yaml_request_template():
    """The first entry in GPT_VARIANTS["gpts"] is what --gpt-variant
    defaults to, and scripts/_gen_module_scripts.py's TEMPLATE treats it as a
    no-op (unsuffixed output filenames, untouched config_dir) -- so it must
    reproduce the real yaml's request_template values verbatim, or a run with
    no --gpt-variant flag would silently diverge from what's actually
    configured."""
    template = yaml.safe_load((ROOT / "configs" / "modules" / "gpts.yaml").read_text())[
        "service"
    ]["request_template"]
    default_overrides = next(iter(GPT_VARIANTS["gpts"].values()))
    for key, value in default_overrides.items():
        assert template[key] == value, f"GPT_VARIANTS['gpts']'s default entry drifted from the YAML for {key!r}"
