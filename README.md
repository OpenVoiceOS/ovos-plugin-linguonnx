# ovos-lang-detect-plugin-lingonnx

An OVOS language-detection plugin. It wraps
[`lingonnx`](https://github.com/TigreGotico/lingonnx), which runs the
[`TigreGotico/glotlid-onnx`](https://huggingface.co/TigreGotico/glotlid-onnx)
export of GlotLID on `onnxruntime`, CPU-only, no torch.

## Install

```bash
pip install ovos-lang-detect-plugin-lingonnx
```

The model downloads from HuggingFace on first use and is cached under
`~/.cache/lingonnx/models/<model_id>/`. Constructing the plugin does not
trigger this download; it happens on the first call to `detect`,
`detect_probs`, or `available_languages`.

## Configuration

```json
{
  "language_detection": {
    "module": "ovos-lang-detect-plugin-lingonnx",
    "ovos-lang-detect-plugin-lingonnx": {
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
- `collapse_varieties`: see below.
- `min_confidence`: if the model's top confidence is below this, `detect()`
  returns `self.config.get("lang", "en-US")` instead of the detected tag.
  Default `0.0`, meaning no fallback ever triggers.
- `lang`: the fallback tag used when `min_confidence` isn't met. Default
  `"en-US"`.

## Language varieties

GlotLID identifies individual language varieties, not macrolanguages.
Casual Arabic text comes back as e.g. `ajp-Arab` (South Levantine) rather
than `ar`, and some Chinese text comes back as `yue-Hani` (Cantonese)
rather than `zh`.

`lingonnx` itself defaults to reporting these varieties as-is, since that
fine-grained answer is useful on its own (free dialect identification).
This plugin flips the default to collapse varieties onto their
macrolanguage (`ajp-Arab` -> `ar`), because the two things OVOS actually
does with a detected language tag - picking a TTS voice, picking a
translation target - only know macrolanguages. A caller getting
`ajp-Arab` back from a pipeline that only ships `ar` voices would read it
as unsupported.

Set `collapse_varieties: false` in the plugin config to get `lingonnx`'s
native per-variety fidelity instead.

## Credits

- [lingonnx](https://github.com/TigreGotico/lingonnx) - the detection engine
  this plugin wraps.
- [TigreGotico/glotlid-onnx](https://huggingface.co/TigreGotico/glotlid-onnx) -
  the ONNX export of GlotLID that `lingonnx` runs.
