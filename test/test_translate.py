import os
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_linguonnx import LinguONNXTranslatePlugin
from ovos_plugin_manager.language import OVOSLangTranslationFactory


def make_mock_translator():
    tx = MagicMock()
    tx.translate.return_value = "bo dia"
    tx.available_languages = frozenset({"en", "pt", "gl", "eu", "ca"})
    tx.can_translate.side_effect = lambda src, tgt: tgt != "eu" or src == "pt"
    return tx


def test_lazy_loading_no_call_on_init():
    with patch("linguonnx.load_translator") as mock_load:
        LinguONNXTranslatePlugin()
        mock_load.assert_not_called()


def test_lazy_loading_calls_once_on_first_use():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        plugin = LinguONNXTranslatePlugin()
        mock_load.assert_not_called()
        plugin.translate("bom dia", "gl", "pt")
        mock_load.assert_called_once()
        plugin.translate("boa tarde", "gl", "pt")
        mock_load.assert_called_once()


def test_translate_passes_src_and_tgt():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        plugin = LinguONNXTranslatePlugin()
        assert plugin.translate("bom dia", "gl", "pt") == "bo dia"
        mock_tx.translate.assert_called_once_with("bom dia", src="pt", tgt="gl")


def test_translate_regional_tags_pass_through_untouched():
    # linguonnx normalizes tags itself; the plugin must not pre-mangle them,
    # or a caller asking for pt-BR silently loses the variety.
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        LinguONNXTranslatePlugin().translate("bom dia", "gl-ES", "pt-BR")
        mock_tx.translate.assert_called_once_with("bom dia", src="pt-BR", tgt="gl-ES")


def test_target_defaults_to_internal_then_lang():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        plugin = LinguONNXTranslatePlugin({"internal": "en", "lang": "pt"})
        plugin.translate("bom dia")
        mock_tx.translate.assert_called_once_with("bom dia", src="pt", tgt="en")


def test_target_falls_back_to_lang_when_no_internal():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        LinguONNXTranslatePlugin({"lang": "ca"}).translate("hola")
        mock_tx.translate.assert_called_once_with("hola", src="ca", tgt="ca")


def test_target_falls_back_to_english_with_empty_config():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        LinguONNXTranslatePlugin().translate("bom dia")
        mock_tx.translate.assert_called_once_with("bom dia", src="en", tgt="en")


def test_no_config_passes_no_loader_kwargs():
    # An unset key must not be defaulted here: linguonnx owns its defaults.
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin().translate("hi", "pt", "en")
        mock_load.assert_called_once_with()


def test_config_passthrough_to_load_translator():
    cfg = {"prefer": "dedicated", "max_hops": 3, "pivot_ranking": "table",
           "include_noncommercial": True, "precision": "fp32",
           "model_cache_size": 2, "num_beams": 1, "max_new_tokens": 256,
           "exclude_flagged": True, "min_chrf": 40.0,
           "models": ["opus-mt-en-pt"]}
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin(dict(cfg)).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(**cfg)


def test_memory_bounding_keys_reach_load_translator():
    """The two keys an operator needs to stop a server being OOM-killed.

    `max_loaded_mb` bounds what the model cache retains;
    `max_concurrent_translations` bounds how many models can be in flight,
    which is the term that actually drives peak memory. Dropping either one
    silently would leave an operator who set it believing it took effect.
    """
    cfg = {"max_loaded_mb": 2000, "max_concurrent_translations": 4}
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin(dict(cfg)).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(**cfg)


def test_generation_and_routing_keys_reach_load_translator():
    cfg = {"pivot_preference": ["en", "es"], "max_routes": 5,
           "length_penalty": 1.2, "no_repeat_ngram_size": 3}
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin(dict(cfg)).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(**cfg)


def test_every_load_translator_parameter_is_forwardable():
    """The whitelist must not silently drop a linguonnx option.

    A hand-maintained tuple drifts the moment linguonnx grows a parameter,
    and the failure is silent: the key is accepted from config and thrown
    away. This compares the two directly so the drift is a test failure
    instead of a support ticket.
    """
    import inspect

    # Deliberately NOT `from linguonnx import load_translator`: the top-level
    # name is a lazy-import wrapper whose signature is bare *args/**kwargs, so
    # reading it would leave `accepted` empty and this test would pass without
    # checking anything. The real function is the contract.
    from linguonnx.translate import load_translator

    plugin = LinguONNXTranslatePlugin()
    accepted = {
        name for name, param in
        inspect.signature(load_translator).parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    }
    assert len(accepted) > 10, \
        f"read a suspiciously bare signature ({sorted(accepted)}); this test " \
        f"would pass vacuously"
    # Every config key the plugin would forward, discovered by offering it
    # every parameter linguonnx accepts.
    plugin.config = {name: object() for name in accepted}
    forwarded = set(plugin.loader_kwargs)
    assert accepted - forwarded == set(), \
        f"load_translator parameters the plugin silently drops: " \
        f"{sorted(accepted - forwarded)}"


def test_unknown_config_keys_are_not_forwarded():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin({"lang": "pt", "module": "x",
                                  "num_beams": 1}).translate("oi", "en")
        mock_load.assert_called_once_with(num_beams=1)


def test_max_model_mb_is_forwarded_to_load_translator():
    # max_model_mb is a routing filter linguonnx's load_translator accepts
    # directly - it must reach load_translator as a kwarg, not become an
    # environment variable (that is the download-budget knob, a different
    # thing).
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin({"max_model_mb": 512}).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(max_model_mb=512)


def test_max_model_mb_does_not_touch_environ():
    # Constructing the plugin must never mutate os.environ - that leaks a
    # per-plugin-instance cap into every other linguonnx consumer sharing the
    # process.
    before = dict(os.environ)
    with patch("linguonnx.load_translator", return_value=make_mock_translator()):
        LinguONNXTranslatePlugin({"max_model_mb": 512}).translate("hi", "pt", "en")
    assert os.environ == before


def test_max_model_mb_unset_is_not_forwarded():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin().translate("hi", "pt", "en")
        mock_load.assert_called_once_with()


# -- the size-cap policy pair ------------------------------------------------
#
# `max_model_mb` alone is not a policy, it is half of one. linguonnx decides
# what the cap MEANS from two more keys, and both were missing from the
# passthrough while sitting in the deployed mycroft.conf on
# translate.openvoiceos.pt - declared, and read by nothing.
#
# The half that bites: `count_cached_as_free` defaults to True whenever
# `oversize_fallback` is not passed. On a host with a pre-warmed cache that
# exempts every cached model from the cap, so a 500 MB cap over a warm 25 GB
# cache filters nothing at all. Dropping either key silently turns the cap
# into a no-op or into a language-deleting filter, and neither failure raises.


def test_oversize_fallback_is_forwarded_to_load_translator():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin({"oversize_fallback": True}).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(oversize_fallback=True)


def test_count_cached_as_free_is_forwarded_to_load_translator():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin({"count_cached_as_free": False}).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(count_cached_as_free=False)


def test_count_cached_as_free_false_is_forwarded_not_dropped_as_falsy():
    """`False` is the value that makes the cap bite; a truthiness test eats it."""
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin({"count_cached_as_free": False}).translate("hi", "pt", "en")
        kwargs = mock_load.call_args.kwargs
        assert "count_cached_as_free" in kwargs
        assert kwargs["count_cached_as_free"] is False


def test_the_deployed_production_config_arrives_intact():
    """The exact `linguonnxsrv/conf/mycroft.conf` from translate.openvoiceos.pt.

    Asserted as one call rather than key by key, because the bug being pinned
    was a whole-policy bug: `max_model_mb` arrived and the two keys that give
    it its meaning did not, which is worse than none of them arriving.
    """
    cfg = {"prefer": "dedicated", "max_hops": 2, "pivot_ranking": "auto",
           "include_noncommercial": False, "precision": "int8",
           "model_cache_size": 4, "num_beams": 4, "max_new_tokens": 512,
           "max_model_mb": 500, "oversize_fallback": True,
           "count_cached_as_free": False}
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin(dict(cfg)).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(**cfg)


def test_the_size_cap_keys_are_not_defaulted_when_unset():
    """linguonnx owns its defaults - especially the `count_cached_as_free`
    tri-state, which picks False under a fallback and True without one. A
    default invented here would overwrite that choice."""
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin({"max_model_mb": 500}).translate("hi", "pt", "en")
        kwargs = mock_load.call_args.kwargs
        assert kwargs == {"max_model_mb": 500}


def test_the_size_cap_keys_do_not_touch_environ():
    """Same guarantee `max_model_mb` already has: no cross-instance leakage."""
    before = dict(os.environ)
    with patch("linguonnx.load_translator", return_value=make_mock_translator()):
        LinguONNXTranslatePlugin({"max_model_mb": 500,
                                  "oversize_fallback": True,
                                  "count_cached_as_free": False}).translate("hi", "pt", "en")
    assert os.environ == before


def test_available_languages_is_a_mutable_set():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        langs = LinguONNXTranslatePlugin().available_languages
        assert langs == {"en", "pt", "gl", "eu", "ca"}
        assert isinstance(langs, set)
        langs.add("fr")  # must not raise; frozenset would


def test_supported_translations_excludes_unroutable_and_self():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        plugin = LinguONNXTranslatePlugin()
        assert plugin.supported_translations("en") == {"pt", "gl", "ca"}
        assert plugin.supported_translations("pt") == {"en", "gl", "eu", "ca"}


def test_supported_translations_is_cached_per_source():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx):
        plugin = LinguONNXTranslatePlugin()
        plugin.supported_translations("en")
        calls = mock_tx.can_translate.call_count
        plugin.supported_translations("en")
        assert mock_tx.can_translate.call_count == calls


def test_entry_point_resolves_to_class():
    from importlib.metadata import entry_points

    eps = entry_points(group="opm.lang.translate")
    matches = [ep for ep in eps if ep.name == "ovos-translate-plugin-linguonnx"]
    assert matches, "entry point ovos-translate-plugin-linguonnx not registered"
    assert matches[0].load() is LinguONNXTranslatePlugin


@pytest.mark.network
def test_real_translation_end_to_end():
    """Downloads and runs a real model. Skipped unless -m network is asked for."""
    plugin = LinguONNXTranslatePlugin({"num_beams": 1})
    out = plugin.translate("Good morning, my friend.", "gl", "en")
    assert isinstance(out, str) and out.strip()
    assert out.strip().lower() != "good morning, my friend."
    assert "gl" in plugin.supported_translations("en")


def test_no_route_becomes_a_valueerror():
    # Callers that only know the OVOS interface cannot import linguonnx's
    # exception types, so an unsupported pair must arrive as a builtin.
    from linguonnx.translate import NoRouteError

    mock_tx = make_mock_translator()
    mock_tx.translate.side_effect = NoRouteError("no route from 'en' to 'xx'")
    with patch("linguonnx.load_translator", return_value=mock_tx):
        with pytest.raises(ValueError, match="no route from 'en' to 'xx'") as caught:
            LinguONNXTranslatePlugin().translate("hi", "xx", "en")
        assert not isinstance(caught.value, NoRouteError)
        assert isinstance(caught.value.__cause__, NoRouteError)


def test_uncached_model_becomes_a_runtimeerror():
    from linguonnx.model_manager import DownloadTooLargeError

    mock_tx = make_mock_translator()
    mock_tx.translate.side_effect = DownloadTooLargeError("needs a 157 MB download")
    with patch("linguonnx.load_translator", return_value=mock_tx):
        with pytest.raises(RuntimeError, match="157 MB") as caught:
            LinguONNXTranslatePlugin().translate("hi", "gl", "en")
        assert not isinstance(caught.value, DownloadTooLargeError)
        assert isinstance(caught.value.__cause__, DownloadTooLargeError)


def test_a_download_error_is_not_reported_as_an_unsupported_pair():
    # RuntimeError, not ValueError: the pair is fine, the host is not. A server
    # mapping ValueError to 4xx must not blame the caller for a cold cache.
    from linguonnx.model_manager import DownloadTooLargeError

    mock_tx = make_mock_translator()
    mock_tx.translate.side_effect = DownloadTooLargeError("cold")
    with patch("linguonnx.load_translator", return_value=mock_tx):
        with pytest.raises(RuntimeError) as caught:
            LinguONNXTranslatePlugin().translate("hi", "gl", "en")
        assert not isinstance(caught.value, ValueError)


# -- documented config section pin -------------------------------------------
#
# ovos-plugin-manager resolves plugin config off a top-level `language`
# section (`OVOSLangTranslationFactory.get_class`/`create`, which read
# `config["language"]`, then `get_plugin_config(config, "language", module)`
# in `ovos_plugin_manager.utils.config`). A doc that names any other section
# key produces settings the plugin never receives.


def test_language_section_config_reaches_load_translator():
    mock_tx = make_mock_translator()
    config = {
        "language": {
            "translation_module": "ovos-translate-plugin-linguonnx",
            "ovos-translate-plugin-linguonnx": {"max_model_mb": 512},
        }
    }
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        plugin = OVOSLangTranslationFactory.create(config)
        plugin.translate("hi", "pt", "en")
        mock_load.assert_called_once_with(max_model_mb=512)


def test_wrong_section_key_never_reaches_load_translator():
    # Negative control: a config nested under `language_translation` (or any
    # key other than `language`) is invisible to the factory. It must not
    # silently produce an unconfigured plugin - it must fail to resolve a
    # module at all.
    config = {
        "language_translation": {
            "module": "ovos-translate-plugin-linguonnx",
            "ovos-translate-plugin-linguonnx": {"max_model_mb": 512},
        }
    }
    with patch("linguonnx.load_translator") as mock_load:
        with pytest.raises(ValueError, match="translation_module"):
            OVOSLangTranslationFactory.create(config)
        mock_load.assert_not_called()
