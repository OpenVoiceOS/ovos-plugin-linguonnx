from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_linguonnx import LinguONNXLangDetectPlugin
from ovos_plugin_linguonnx.detect import DEFAULT_MODEL


def make_mock_detector():
    det = MagicMock()
    det.detect_raw.return_value = ("por_Latn", 0.9)
    det.detect.return_value = "pt"
    det._label_mapper.to_bcp47.return_value = "pt"
    det.detect_probs.return_value = {"pt": 0.9, "gl": 0.05}
    det.available_languages = {"pt", "en", "gl"}
    return det


def test_lazy_loading_no_call_on_init():
    with patch("linguonnx.load_detector") as mock_load:
        LinguONNXLangDetectPlugin()
        mock_load.assert_not_called()


def test_lazy_loading_calls_on_first_use():
    mock_det = make_mock_detector()
    with patch("linguonnx.load_detector", return_value=mock_det) as mock_load:
        plugin = LinguONNXLangDetectPlugin()
        mock_load.assert_not_called()
        plugin.detect("bom dia")
        mock_load.assert_called_once_with(DEFAULT_MODEL)
        # second call must not reload
        plugin.detect("bom dia outra vez")
        mock_load.assert_called_once()


def test_default_config():
    plugin = LinguONNXLangDetectPlugin()
    assert plugin.model == DEFAULT_MODEL
    assert plugin.collapse_varieties is True
    assert plugin.min_confidence == 0.0


def test_collapse_varieties_default_true_collapses_variety():
    # South Levantine Arabic (ajp) collapses onto the macrolanguage "ar" when
    # collapse_varieties is on (the default).
    mock_det = make_mock_detector()
    mock_det.detect_raw.return_value = ("ajp_Arab", 0.9)
    mock_det._label_mapper.to_bcp47.return_value = "ajp-Arab"
    with patch("linguonnx.load_detector", return_value=mock_det):
        plugin = LinguONNXLangDetectPlugin()
        result = plugin.detect("some arabic text")
        assert result == "ar"
        mock_det._label_mapper.to_bcp47.assert_called_once_with("ajp_Arab")


def test_collapse_varieties_false_keeps_variety():
    mock_det = make_mock_detector()
    mock_det.detect_raw.return_value = ("ajp_Arab", 0.9)
    mock_det._label_mapper.to_bcp47.return_value = "ajp-Arab"
    with patch("linguonnx.load_detector", return_value=mock_det):
        plugin = LinguONNXLangDetectPlugin({"collapse_varieties": False})
        result = plugin.detect("some arabic text")
        assert result == "ajp-Arab"


def test_detect_runs_inference_exactly_once():
    # detect_raw() alone runs the ONNX forward pass; linguonnx's own detect()
    # would run it again internally for the same answer, doubling inference
    # cost for zero new information. detect() must never be called.
    mock_det = make_mock_detector()
    with patch("linguonnx.load_detector", return_value=mock_det):
        LinguONNXLangDetectPlugin().detect("bom dia")
        mock_det.detect_raw.assert_called_once_with("bom dia")
        mock_det.detect.assert_not_called()


def test_min_confidence_fallback():
    mock_det = make_mock_detector()
    mock_det.detect_raw.return_value = ("por_Latn", 0.2)
    with patch("linguonnx.load_detector", return_value=mock_det):
        plugin = LinguONNXLangDetectPlugin({"min_confidence": 0.5})
        result = plugin.detect("ambiguous text")
        assert result == "en-US"
        mock_det.detect.assert_not_called()


def test_min_confidence_fallback_uses_configured_lang():
    mock_det = make_mock_detector()
    mock_det.detect_raw.return_value = ("por_Latn", 0.2)
    with patch("linguonnx.load_detector", return_value=mock_det):
        plugin = LinguONNXLangDetectPlugin({"min_confidence": 0.5, "lang": "es-ES"})
        result = plugin.detect("ambiguous text")
        assert result == "es-ES"


def test_min_confidence_passes_through_above_threshold():
    mock_det = make_mock_detector()
    with patch("linguonnx.load_detector", return_value=mock_det):
        plugin = LinguONNXLangDetectPlugin({"min_confidence": 0.5})
        result = plugin.detect("bom dia")
        assert result == "pt"


def test_detect_probs_shape():
    mock_det = make_mock_detector()
    with patch("linguonnx.load_detector", return_value=mock_det):
        plugin = LinguONNXLangDetectPlugin()
        probs = plugin.detect_probs("bom dia")
        assert isinstance(probs, dict)
        assert probs == {"pt": 0.9, "gl": 0.05}
        for k, v in probs.items():
            assert isinstance(k, str)
            assert isinstance(v, float)


def test_available_languages():
    mock_det = make_mock_detector()
    with patch("linguonnx.load_detector", return_value=mock_det):
        plugin = LinguONNXLangDetectPlugin()
        assert plugin.available_languages == {"pt", "en", "gl"}


def test_custom_model_config():
    mock_det = make_mock_detector()
    with patch("linguonnx.load_detector", return_value=mock_det) as mock_load:
        plugin = LinguONNXLangDetectPlugin({"model": "glotlid"})
        plugin.detect("bom dia")
        mock_load.assert_called_once_with("glotlid")


def test_entry_point_resolves_to_class():
    from importlib.metadata import entry_points

    eps = entry_points(group="opm.lang.detect")
    matches = [ep for ep in eps if ep.name == "ovos-lang-detect-plugin-linguonnx"]
    assert matches, "entry point ovos-lang-detect-plugin-linguonnx not registered"
    cls = matches[0].load()
    assert cls is LinguONNXLangDetectPlugin
