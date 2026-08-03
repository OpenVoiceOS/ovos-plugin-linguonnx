import os
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_linguonnx import LinguONNXTranslatePlugin


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
           "model_cache_size": 2, "num_beams": 1, "max_new_tokens": 256}
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin(dict(cfg)).translate("hi", "pt", "en")
        mock_load.assert_called_once_with(**cfg)


def test_unknown_config_keys_are_not_forwarded():
    mock_tx = make_mock_translator()
    with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
        LinguONNXTranslatePlugin({"lang": "pt", "module": "x",
                                  "num_beams": 1}).translate("oi", "en")
        mock_load.assert_called_once_with(num_beams=1)


def test_max_model_mb_sets_env_and_is_not_a_loader_kwarg():
    mock_tx = make_mock_translator()
    old = os.environ.pop("LINGUONNX_MAX_DOWNLOAD_MB", None)
    try:
        with patch("linguonnx.load_translator", return_value=mock_tx) as mock_load:
            LinguONNXTranslatePlugin({"max_model_mb": 512}).translate("hi", "pt", "en")
            mock_load.assert_called_once_with()
            assert os.environ["LINGUONNX_MAX_DOWNLOAD_MB"] == "512"
    finally:
        os.environ.pop("LINGUONNX_MAX_DOWNLOAD_MB", None)
        if old is not None:
            os.environ["LINGUONNX_MAX_DOWNLOAD_MB"] = old


def test_max_model_mb_unset_leaves_env_alone():
    mock_tx = make_mock_translator()
    os.environ.pop("LINGUONNX_MAX_DOWNLOAD_MB", None)
    with patch("linguonnx.load_translator", return_value=mock_tx):
        LinguONNXTranslatePlugin().translate("hi", "pt", "en")
    assert "LINGUONNX_MAX_DOWNLOAD_MB" not in os.environ


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
