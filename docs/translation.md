# Translation

`LinguONNXTranslatePlugin` (entry point `opm.lang.translate`, plugin id
`ovos-translate-plugin-linguonnx`) wraps `linguonnx.translate` as an OVOS
translation plugin. For the full config reference see
[`configuration.md`](configuration.md).

## Routing

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

How a route is scored — the `prefer` policies, hop caps, the size budget,
pivot ranking, pinning a route yourself — is covered in full by `linguonnx`'s
own [`docs/routing.md`](https://github.com/OpenVoiceOS/linguonnx/blob/dev/docs/routing.md).
The size and download-budget interaction that most often surprises a
deployment is documented in
[Size and fallback routing](configuration.md#size-and-fallback-routing).

## Prefetching

On a server, no request should ever trigger a download. Warm the cache at
deploy time:

```python
from linguonnx import load_translator
from linguonnx.model_manager import prefetch

prefetch(*load_translator().models, kind="translate")
```

The default permissive int8 selection is 73 models, about 25 GB on disk.
