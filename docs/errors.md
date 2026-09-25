# Errors

`translate()` raises builtins, never a `linguonnx` exception type, so a caller
that only knows the OVOS interface can still tell the two failure modes apart:

- **`ValueError`** — the pair is not routable. The caller asked for something
  this graph cannot serve. `linguonnx`'s own message says which bound blocked
  it (hop cap, licence filter, unrunnable model), so it is passed through.
- **`RuntimeError`** — the pair is routable, but the model is not on disk and
  the cold-download budget (`LINGUONNX_MAX_DOWNLOAD_MB`, default 8192 MB)
  refused to fetch it on the request path. This is a different bound from
  `max_model_mb` (see [configuration.md](configuration.md)) — a route can be
  planned within `max_model_mb` and still hit this if the model was never
  downloaded. Retrying will not help; prefetch the model
  (see [Prefetching](translation.md#prefetching)).

A server in front of this should map the first to 4xx and the second to 5xx.
Blaming the caller for a cold cache is the mistake worth avoiding.
