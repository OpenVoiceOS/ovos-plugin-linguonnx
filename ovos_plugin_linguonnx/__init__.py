"""OVOS language plugins backed by `linguonnx <https://github.com/OpenVoiceOS/linguonnx>`_.

One package ships both halves of the library:

``LinguONNXLangDetectPlugin``
    ``opm.lang.detect`` - GlotLID language identification over ONNX Runtime.
``LinguONNXTranslatePlugin``
    ``opm.lang.translate`` - offline machine translation over the linguonnx
    routing graph (M2M100 plus the opus-mt bilingual pairs by default).

Both load their models on first use, so importing or constructing either class
never touches the disk or the network.
"""

from ovos_plugin_linguonnx.detect import LinguONNXLangDetectPlugin
from ovos_plugin_linguonnx.translate import LinguONNXTranslatePlugin
from ovos_plugin_linguonnx.version import __version__

# Historical aliases. `ovos-lang-detect-plugin-linguonnx` exported
# `LanguageDetectorPlugin` from its top-level package, and configs and imports
# in the wild still use it.
LanguageDetectorPlugin = LinguONNXLangDetectPlugin
LanguageTranslatorPlugin = LinguONNXTranslatePlugin

__all__ = ["LinguONNXLangDetectPlugin", "LinguONNXTranslatePlugin",
           "LanguageDetectorPlugin", "LanguageTranslatorPlugin", "__version__"]
