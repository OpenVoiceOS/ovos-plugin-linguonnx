# Language detection

`LinguONNXLangDetectPlugin` (entry point `opm.lang.detect`, plugin id
`ovos-lang-detect-plugin-linguonnx`) wraps `linguonnx.detect` as an OVOS
language-detection plugin.

```json
{
  "language": {
    "detection_module": "ovos-lang-detect-plugin-linguonnx",
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
- `collapse_varieties`: see [Language varieties](#language-varieties) below.
- `min_confidence`: if the model's top confidence is below this, `detect()`
  returns `self.config.get("lang", "en-US")` instead of the detected tag.
  Default `0.0`, meaning no fallback ever triggers.
- `lang`: the fallback tag used when `min_confidence` isn't met. Default
  `"en-US"`.

For the models themselves, BCP-47 label mapping, hierarchical softmax, short
text reliability and input limits, see `linguonnx`'s own
[`docs/detect.md`](https://github.com/OpenVoiceOS/linguonnx/blob/dev/docs/detect.md).

## Language varieties

GlotLID identifies individual language varieties, not macrolanguages.
Casual Arabic text comes back as e.g. `ajp-Arab` (South Levantine) rather
than `ar`, and some Chinese text comes back as `yue-Hani` (Cantonese)
rather than `zh`.

`collapse_varieties` selects which of the two tags `detect()` returns.

| Setting | South Levantine Arabic | Cantonese |
|---|---|---|
| `true` (default) | `ar` | `zh-Hani` |
| `false` | `ajp-Arab` | `yue-Hani` |

### Why the default collapses

OVOS matches a language tag by `langcodes` tag distance, and it accepts a
candidate below distance 10. `closest_lang` in `ovos-spec-tools` implements
that rule. `disambiguate_lang` in `ovos-core` applies it to the
`detected_lang` context key this plugin writes. `ovos-workshop` applies it to
locale directories and to converse matchers.

The threshold covers region and script subtags. `ar-SA` matches `ar` at
distance 4. `pt-BR` matches `pt-PT` at distance 5. A regional dialect
therefore needs no help from this plugin.

The threshold does not cover a member of a macrolanguage, because `langcodes`
rates such a member as a separate language. `ajp` is at distance 10 from `ar`.
`arz` is at distance 10 from `ar`. `yue` is at distance 64 from `zh`. Of the
47 varieties `linguonnx` maps to a macrolanguage, 27 stay at or above the
threshold. For those, a variety tag matches nothing. OVOS then discards the
detection, and the session keeps its previous language.
`ovos-bidirectional-translation-plugin` is stricter again. It tests the
detected tag for exact membership of the configured languages, so any variety
tag fails there.

### When to keep the variety

Set `collapse_varieties: false` when the caller reads the dialect itself, for
example to log it or to route it. This setting keeps information that the
default discards. The caller then owns the match against the languages it
supports.
