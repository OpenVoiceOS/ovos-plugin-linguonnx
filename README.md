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
      "exclude_flagged": false,
      "min_chrf": null,
      "models": null,
      "model_cache_size": 4,
      "max_model_mb": null,
      "oversize_fallback": false,
      "count_cached_as_free": null,
      "num_beams": 4,
      "max_new_tokens": 128
    }
  }
}
```

Every key is optional. A key you leave out is not defaulted by the plugin — it
is simply not passed, so `linguonnx` stays the single source of truth for what
a default is. The values shown above for `max_model_mb`, `oversize_fallback`
and `count_cached_as_free` are "unset" / library defaults, not numbers to copy
in — see the worked example under
[Size and fallback routing](#size-and-fallback-routing) for what a real
deployment sets.

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
- `exclude_flagged`: drop any model linguonnx's own quality sweep flags —
  either precision scoring below 40 chrF against the FLORES-200 reference, or
  int8 trailing fp32 by more than 2 chrF. Combine with `precision: null` to
  fall back to fp32 wherever int8 alone is flagged.
- `min_chrf`: drop any model scoring below this chrF against FLORES-200.
  `null` keeps every model the other filters allow through.
- `models`: exact registry ids to route over, overriding every other filter
  above. A list of model ids as linguonnx names them.
- `model_cache_size`: how many loaded models stay in memory at once,
  least-recently-used evicted first. The whole default graph is ~25 GB, so
  this is the knob that keeps a long-lived server from being OOM-killed.
- `max_model_mb`: drop any model bigger than this, in MB, from the routing
  graph before a route is even scored — this is a **routing** filter, not a
  download guard. Unset by default, which leaves every model in the graph
  eligible (in practice this still ends up bounded by the cold-download
  budget — see [Size and fallback routing](#size-and-fallback-routing)
  below). Combining this with `models` makes `linguonnx` treat it as an
  operator-set budget that outranks the `models` waiver; see the `linguonnx`
  docs before setting both.
- `oversize_fallback`: `false` by default, which makes `max_model_mb` a
  hard filter — a language that lives only inside an oversized model becomes
  unroutable. `true` makes the cap a *preference* instead: every pair a model
  under the cap can serve is still served by that model, and only a pair
  nothing under the cap covers escalates to the smallest oversized model that
  does. See [Size and fallback routing](#size-and-fallback-routing).
- `count_cached_as_free`: whether a model already in the local cache is
  exempt from `max_model_mb`. Left unset, `linguonnx` picks `true` normally
  and `false` when `oversize_fallback` is on, because a warm cache would
  otherwise exempt every model there is and make the cap a no-op. Set it
  explicitly only to overrule that: `true` reads the cap as "do not download
  more than this", `false` as "do not load a model bigger than this".
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

### Size and fallback routing

`max_model_mb` bounds the size of a single model a route may use. On its own
it is a hard filter: a pair only a big multilingual model covers becomes
unroutable once that model is over the cap. `oversize_fallback: true` turns
it into a preference — a pair a small model can serve still gets that model,
and only a pair *nothing* under the cap serves escalates to the smallest
oversized model that does, reporting `waived_size_cap` on the returned route
so the exception is visible rather than silent.

A deployment tuned for load latency, not disk space, looks like this:

```json
{
  "language_translation": {
    "module": "ovos-translate-plugin-linguonnx",
    "ovos-translate-plugin-linguonnx": {
      "precision": "int8",
      "max_model_mb": 500,
      "oversize_fallback": true,
      "count_cached_as_free": false
    }
  }
}
```

Measured against the default registry: `max_model_mb: 500` keeps roughly
fifteen warm 1.7–2.0 GB multilingual models (NLLB, M2M100, and friends) from
winning a route even though they are already on disk — `count_cached_as_free:
false` is what makes that true; without it a warm cache would exempt those
models from the cap entirely and the 500 MB number would do nothing.
`en -> ca` then routes through a 157 MB `opus-mt` model instead of a bigger
multilingual one. `en -> cv` (Chuvash) has no model under 500 MB that covers
it at all, so `oversize_fallback: true` escalates to the smallest oversized
model that does — a 4945 MB MADLAD checkpoint — rather than failing. Both
outcomes are correct for this config; 586 languages stay routable across the
whole graph either way, versus the 249 you get if `oversize_fallback` is left
`false` and the cap simply deletes everything above it.

**Do not set `LINGUONNX_MAX_DOWNLOAD_MB` at or near `max_model_mb`.** It is a
separate environment variable — a cold-download ceiling, not a routing knob —
and it caps how far `oversize_fallback` is allowed to escalate: the fallback
never admits a model the downloader would then refuse. Setting it to `500`
alongside a `max_model_mb: 500` routing cap collapses the fallback entirely,
because nothing can escalate past a ceiling equal to the cap itself — routable
languages on the default registry drop from 586 to 249, the same as having no
fallback at all. Leave it at its default (8192 MB) unless you deliberately
want to bound how large a model `oversize_fallback` may fetch.

Routing is also **cache-dependent** whenever the effective download budget is
below the size of the largest runnable model: the same registry and the same
config can produce different routes on a cold host versus one with the big
models already downloaded, because a cached model costs no download and the
budget never sees it. This is expected, not a bug — see `linguonnx`'s own
[`docs/routing.md`](https://github.com/OpenVoiceOS/linguonnx/blob/dev/docs/routing.md#routing-is-cache-dependent)
for the full explanation and measured before/after numbers. Prefetch the
models you intend to serve (see [Prefetching](#prefetching)) if you need a
fleet of hosts to agree on the same routes.

### Errors

`translate()` raises builtins, never a `linguonnx` exception type, so a caller
that only knows the OVOS interface can still tell the two failure modes apart:

- **`ValueError`** — the pair is not routable. The caller asked for something
  this graph cannot serve. `linguonnx`'s own message says which bound blocked
  it (hop cap, licence filter, unrunnable model), so it is passed through.
- **`RuntimeError`** — the pair is routable, but the model is not on disk and
  the cold-download budget (`LINGUONNX_MAX_DOWNLOAD_MB`, default 8192 MB)
  refused to fetch it on the request path. This is a different bound from
  `max_model_mb` above — a route can be planned within `max_model_mb` and
  still hit this if the model was never downloaded. Retrying will not help;
  prefetch the model.

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
