from unittest.mock import MagicMock, patch

import pytest

from ovos_lang_detect_plugin_lingonnx import LingoNNXLangDetectPlugin, DEFAULT_MODEL


def make_mock_detector():
    det = MagicMock()
    det.detect_raw.return_value = ("por_Latn", 0.9)
    det.detect.return_value = "pt"
    det.detect_probs.return_value = {"pt": 0.9, "gl": 0.05}
    det.available_languages = {"pt", "en", "gl"}
    return det


def test_lazy_loading_no_call_on_init():
    with patch("lingonnx.load_detector") as mock_load:
        LingoNNXLangDetectPlugin()
        mock_load.assert_not_called()


def test_lazy_loading_calls_on_first_use():
    mock_det = make_mock_detector()
    with patch("lingonnx.load_detector", return_value=mock_det) as mock_load:
        plugin = LingoNNXLangDetectPlugin()
        mock_load.assert_not_called()
        plugin.detect("bom dia")
        mock_load.assert_called_once_with(DEFAULT_MODEL)
        # second call must not reload
        plugin.detect("bom dia outra vez")
        mock_load.assert_called_once()


def test_default_config():
    plugin = LingoNNXLangDetectPlugin()
    assert plugin.model == DEFAULT_MODEL
    assert plugin.collapse_varieties is True
    assert plugin.min_confidence == 0.0


def test_collapse_varieties_default_true_passthrough():
    mock_det = make_mock_detector()
    with patch("lingonnx.load_detector", return_value=mock_det):
        plugin = LingoNNXLangDetectPlugin()
        plugin.detect("some arabic text")
        mock_det.detect.assert_called_once_with("some arabic text", collapse_varieties=True)


def test_collapse_varieties_false_passthrough():
    mock_det = make_mock_detector()
    with patch("lingonnx.load_detector", return_value=mock_det):
        plugin = LingoNNXLangDetectPlugin({"collapse_varieties": False})
        plugin.detect("some arabic text")
        mock_det.detect.assert_called_once_with("some arabic text", collapse_varieties=False)


def test_min_confidence_fallback():
    mock_det = make_mock_detector()
    mock_det.detect_raw.return_value = ("por_Latn", 0.2)
    with patch("lingonnx.load_detector", return_value=mock_det):
        plugin = LingoNNXLangDetectPlugin({"min_confidence": 0.5})
        result = plugin.detect("ambiguous text")
        assert result == "en-US"
        mock_det.detect.assert_not_called()


def test_min_confidence_fallback_uses_configured_lang():
    mock_det = make_mock_detector()
    mock_det.detect_raw.return_value = ("por_Latn", 0.2)
    with patch("lingonnx.load_detector", return_value=mock_det):
        plugin = LingoNNXLangDetectPlugin({"min_confidence": 0.5, "lang": "es-ES"})
        result = plugin.detect("ambiguous text")
        assert result == "es-ES"


def test_min_confidence_passes_through_above_threshold():
    mock_det = make_mock_detector()
    with patch("lingonnx.load_detector", return_value=mock_det):
        plugin = LingoNNXLangDetectPlugin({"min_confidence": 0.5})
        result = plugin.detect("bom dia")
        assert result == "pt"


def test_detect_probs_shape():
    mock_det = make_mock_detector()
    with patch("lingonnx.load_detector", return_value=mock_det):
        plugin = LingoNNXLangDetectPlugin()
        probs = plugin.detect_probs("bom dia")
        assert isinstance(probs, dict)
        assert probs == {"pt": 0.9, "gl": 0.05}
        for k, v in probs.items():
            assert isinstance(k, str)
            assert isinstance(v, float)


def test_available_languages():
    mock_det = make_mock_detector()
    with patch("lingonnx.load_detector", return_value=mock_det):
        plugin = LingoNNXLangDetectPlugin()
        assert plugin.available_languages == {"pt", "en", "gl"}


def test_custom_model_config():
    mock_det = make_mock_detector()
    with patch("lingonnx.load_detector", return_value=mock_det) as mock_load:
        plugin = LingoNNXLangDetectPlugin({"model": "glotlid"})
        plugin.detect("bom dia")
        mock_load.assert_called_once_with("glotlid")


def test_entry_point_resolves_to_class():
    from importlib.metadata import entry_points

    eps = entry_points(group="opm.lang.detect")
    matches = [ep for ep in eps if ep.name == "ovos-lang-detect-plugin-lingonnx"]
    assert matches, "entry point ovos-lang-detect-plugin-lingonnx not registered"
    cls = matches[0].load()
    assert cls is LingoNNXLangDetectPlugin
