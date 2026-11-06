import pytest

from moe_interp.config import ConfigError, apply_overrides, config_from_dict, load_config, save_config


def test_defaults_and_repo_configs():
    cfg = load_config(None)
    assert cfg.generation.do_sample is False and cfg.generation.max_new_tokens == 128
    assert cfg.trace.activations.positions == "last"
    assert cfg.trace.routing.save_router_logits is False
    for name in ("olmoe", "olmoe_full_capture", "debug_tiny"):
        load_config(f"configs/{name}.yaml")


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError, match="max_new_token"):
        config_from_dict({"generation": {"max_new_token": 3}})
    with pytest.raises(ConfigError, match="activation type"):
        config_from_dict({"trace": {"activations": {"types": ["resid"]}}})


def test_overrides_and_round_trip(tmp_path):
    cfg = apply_overrides(load_config(None), ["generation.max_new_tokens=7", "trace.activations.layers=[1, -1]",
                                               "analysis.control=null", "model.model_id=foo/bar"])
    assert cfg.generation.max_new_tokens == 7 and cfg.trace.activations.layers == [1, -1]
    assert cfg.analysis.control is None and cfg.model.model_id == "foo/bar"
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["nope.x=1"])
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["generation"])
