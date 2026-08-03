# ovos-plugin-linguonnx

OVOS language plugins built on
[`linguonnx`](https://github.com/OpenVoiceOS/linguonnx). One package, two
plugins, both CPU-only on `onnxruntime`, both fully offline after the models
are cached, neither needs torch:

| Plugin | Entry point group | Plugin id |
|---|---|---|
| `LinguONNXLangDetectPlugin` | `opm.lang.detect` | `ovos-lang-detect-plugin-linguonnx` |
| `LinguONNXTranslatePlugin` | `opm.lang.translate` | `ovos-translate-plugin-linguonnx` |

> This package was called `ovos-lang-detect-plugin-linguonnx` while it only did
> detection. The plugin **ids** did not change, so existing configuration keeps
> working; only the distribution name and the Python package did.

## Install

```bash
pip install ovos-plugin-linguonnx
```

Models download from HuggingFace on first use and are cached under
`~/.cache/linguonnx/models/<model_id>/`. Constructing either plugin does not
trigger a download; the first real call does. On a server, prefetch instead —
see [Prefetching](#prefetching).

## Language detection

```json
{
  "language_detection": {
    "module": "ovos-lang-detect-plugin-linguonnx",
    "ovos-lang-detect-plugin-linguonnx": {
      "model": "glotlid-int8",
      "collapse_varieties": true,
      "min_confidence": 0.0,
      "lang": "en-US"
    }
  }
}
```

- `model`: `"glotlid-int8"` (default, quantized, 419 MB) or `"glotlid"`
  (fp32, 1.68 GB). Both cover the same 2102 GlotLID labels; the int8 model
  has no measured accuracy loss, so there is no reason to prefer the fp32
  one unless you need it for some other purpose.
- `collapse_varieties`: see [Language varieties](#language-varieties).
- `min_confidence`: if the model's top confidence is below this, `detect()`
  returns `self.config.get("lang", "en-US")` instead of the detected tag.
  Default `0.0`, meaning no fallback ever triggers.
- `lang`: the fallback tag used when `min_confidence` isn't met. Default
  `"en-US"`.

### Language varieties

GlotLID identifies individual language varieties, not macrolanguages.
Casual Arabic text comes back as e.g. `ajp-Arab` (South Levantine) rather
than `ar`, and some Chinese text comes back as `yue-Hani` (Cantonese)
rather than `zh`.

`linguonnx` itself defaults to reporting these varieties as-is, since that
fine-grained answer is useful on its own (free dialect identification).
This plugin flips the default to collapse varieties onto their
macrolanguage (`ajp-Arab` -> `ar`), because the two things OVOS actually
does with a detected language tag - picking a TTS voice, picking a
translation target - only know macrolanguages. A caller getting
`ajp-Arab` back from a pipeline that only ships `ar` voices would read it
as unsupported.

Set `collapse_varieties: false` in the plugin config to get `linguonnx`'s
native per-variety fidelity instead.

## Translation

```json
{
  "language_translation": {
    "module": "ovos-translate-plugin-linguonnx",
    "ovos-translate-plugin-linguonnx": {
      "prefer": "fewest_hops",
      "max_hops": 2,
      "pivot_ranking": "auto",
      "include_noncommercial": false,
      "precision": "int8",
      "model_cache_size": 4,
      "max_model_mb": 8192,
      "num_beams": 4,
      "max_new_tokens": 128
    }
  }
}
```

Every key is optional. A key you leave out is not defaulted by the plugin — it
is simply not passed, so `linguonnx` stays the single source of truth for what
a default is. The values above are the `linguonnx` defaults.

- `prefer`: route ranking. `"fewest_hops"` takes the shortest route.
  `"dedicated"` prefers a bilingual model over a multilingual one even when
  that costs an extra hop.
- `max_hops`: how many models a route may chain. `1` is direct models only,
  `2` allows one pivot language. Higher is allowed, but each hop compounds the
  previous hop's errors.
- `pivot_ranking`: how pivot candidates are ordered on a two-hop route.
  `"auto"` uses phonological distance when `orthography2ipa` is installed
  (`pip install linguonnx[distance]`) and a curated table otherwise.
  `"phonological"` demands the package; `"table"` ignores it.
- `include_noncommercial`: `false` uses only permissively licensed models.
  `true` adds NLLB-200 (CC-BY-NC-4.0) — broader coverage, but it puts a
  non-commercial licence on your output.
- `precision`: `"int8"` for the quantized models, `"fp32"` for the full ones
  (roughly 4x the disk and memory), or `null` for both.
- `model_cache_size`: how many loaded models stay in memory at once,
  least-recently-used evicted first. The whole default graph is ~25 GB, so
  this is the knob that keeps a long-lived server from being OOM-killed.
- `max_model_mb`: refuse a cold download larger than this, in MB, instead of
  holding the request for minutes. Set it low on a host whose cache is
  pre-warmed, so a surprise fetch fails fast and loudly. It maps to the
  `LINGUONNX_MAX_DOWNLOAD_MB` environment variable.
- `num_beams`: beam width. `1` is greedy and about 4x faster.
- `max_new_tokens`: output length cap per hop. Raise it for long input; the
  decode loop stops at the cap without warning.

### Routing

`linguonnx` is not one giant multilingual model. It is a graph of models, and a
request is routed through one or two of them. A pair with a direct model
(`en->gl`) runs that model; a pair without one (`pt->eu`) pivots
(`pt->en->eu`). This is why `supported_translations(source)` is narrower than
`available_languages`: a tag can be in the graph and still have no route from
your source.

```python
from ovos_plugin_linguonnx import LinguONNXTranslatePlugin

tx = LinguONNXTranslatePlugin()
tx.translate("Good morning", target="gl", source="en")
tx.supported_translations("en")   # targets routable from English
tx.available_languages            # every tag anywhere in the graph
```

### Errors

`translate()` raises builtins, never a `linguonnx` exception type, so a caller
that only knows the OVOS interface can still tell the two failure modes apart:

- **`ValueError`** — the pair is not routable. The caller asked for something
  this graph cannot serve. `linguonnx`'s own message says which bound blocked
  it (hop cap, licence filter, unrunnable model), so it is passed through.
- **`RuntimeError`** — the pair is routable, but the model is not on disk and
  `max_model_mb` refused to fetch it on the request path. Retrying will not
  help; prefetch the model.

A server in front of this should map the first to 4xx and the second to 5xx.
Blaming the caller for a cold cache is the mistake worth avoiding.

### Prefetching

On a server, no request should ever trigger a download. Warm the cache at
deploy time:

```python
from linguonnx import load_translator
from linguonnx.model_manager import prefetch

prefetch(*load_translator().models, kind="translate")
```

The default permissive int8 selection is 73 models, about 25 GB on disk.

## Credits

- [linguonnx](https://github.com/OpenVoiceOS/linguonnx) — the detection and
  translation engine this package wraps.
- [TigreGotico/glotlid-onnx](https://huggingface.co/TigreGotico/glotlid-onnx) —
  the ONNX export of GlotLID that `linguonnx` runs for detection.
