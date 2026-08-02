"""OVOS language-detection plugin wrapping the linguonnx GlotLID detector."""

from typing import Dict, Optional, Set

from ovos_plugin_manager.templates.language import LanguageDetector

from ovos_lang_detect_plugin_linguonnx.version import __version__

DEFAULT_MODEL = "glotlid-int8"


class LinguONNXLangDetectPlugin(LanguageDetector):
    """LanguageDetector implementation backed by ``linguonnx`` (GlotLID/ONNX).

    The underlying model is only loaded on first call to :meth:`detect`,
    :meth:`detect_probs`, :meth:`detect_raw` or :attr:`available_languages` -
    constructing this class never triggers a model load/download.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._detector = None

    @property
    def model(self) -> str:
        return self.config.get("model", DEFAULT_MODEL)

    @property
    def collapse_varieties(self) -> bool:
        # linguonnx itself defaults this to False (fidelity: "ajp-Arab" is a
        # real, useful answer on its own). OVOS callers use the detected tag
        # to pick a TTS voice or a translation target, and those pipelines
        # only know macrolanguages - "ajp-Arab" would look unsupported where
        # "ar" is. So the plugin flips the default to True; set
        # collapse_varieties: false in config to get linguonnx's raw fidelity.
        return self.config.get("collapse_varieties", True)

    @property
    def min_confidence(self) -> float:
        return self.config.get("min_confidence", 0.0)

    @property
    def detector(self):
        """Lazily load (and cache) the linguonnx detector instance."""
        if self._detector is None:
            from linguonnx import load_detector
            self._detector = load_detector(self.model)
        return self._detector

    def detect(self, text: str) -> str:
        _, confidence = self.detector.detect_raw(text)
        if confidence < self.min_confidence:
            return self.config.get("lang", "en-US")
        return self.detector.detect(text, collapse_varieties=self.collapse_varieties)

    def detect_probs(self, text: str) -> Dict[str, float]:
        return self.detector.detect_probs(text)

    @property
    def available_languages(self) -> Set[str]:
        return self.detector.available_languages


LanguageDetectorPlugin = LinguONNXLangDetectPlugin

__all__ = ["LinguONNXLangDetectPlugin", "LanguageDetectorPlugin", "__version__"]
