# ovos-plugin-linguonnx

OVOS language plugins built on
[`linguonnx`](https://github.com/OpenVoiceOS/linguonnx). One package, two
plugins, both CPU-only on `onnxruntime`, both fully offline after the models
are cached, neither needs torch:

| Plugin | Entry point group | Plugin id |
|---|---|---|
| `LinguONNXLangDetectPlugin` | `opm.lang.detect` | `ovos-lang-detect-plugin-linguonnx` |
| `LinguONNXTranslatePlugin` | `opm.lang.translate` | `ovos-translate-plugin-linguonnx` |

## Install

```bash
pip install ovos-plugin-linguonnx
```

Models download from HuggingFace on first use and are cached under
`~/.cache/linguonnx/models/<model_id>/`. Constructing either plugin does not
trigger a download; the first real call does. On a server, prefetch instead —
see [Prefetching](docs/translation.md#prefetching).

## Usage

```python
from ovos_plugin_linguonnx import LinguONNXLangDetectPlugin, LinguONNXTranslatePlugin

det = LinguONNXLangDetectPlugin()
det.detect("bom dia")   # 'pt'

tx = LinguONNXTranslatePlugin()
tx.translate("Good morning", target="gl", source="en")
```

## Documentation

- [docs/language-detection.md](docs/language-detection.md) — detection
  config, model choice, and language varieties.
- [docs/translation.md](docs/translation.md) — translation config, routing,
  and prefetching.
- [docs/configuration.md](docs/configuration.md) — the full translation
  config reference, including the size/fallback routing budget.
- [docs/docker.md](docs/docker.md) — running both plugins as a server image.
- [docs/deployment.md](docs/deployment.md) — cache mounts, prefetching, memory
  sizing, and the verification checklist for a production service.
- [docs/errors.md](docs/errors.md) — what `translate()` raises and how to
  map it to HTTP status codes.

## Credits

- [linguonnx](https://github.com/OpenVoiceOS/linguonnx) — the detection and
  translation engine this package wraps.
- [TigreGotico/glotlid-onnx](https://huggingface.co/TigreGotico/glotlid-onnx) —
  the ONNX export of GlotLID that `linguonnx` runs for detection.
