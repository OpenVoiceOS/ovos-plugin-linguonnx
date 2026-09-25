# Deployment

Running these plugins as a public or long-lived service is a memory-sizing
problem before it is anything else. The routing graph spans 586 languages and
roughly 109 GB of int8 weights; nothing bounds how much of that a busy server
pulls into RAM unless you say so. This page covers the settings that keep a
server inside a known memory ceiling, and how to size that ceiling.

For the routing knobs themselves — `max_model_mb`, `oversize_fallback`,
`precision` — see [configuration.md](configuration.md). For the container, see
[docker.md](docker.md).

## The three memory bounds

Memory is held in two distinct places, and they need separate bounds.

| Key | Bounds | Default |
|---|---|---|
| `model_cache_size` | How many models the cache retains | 4 |
| `max_loaded_mb` | How many megabytes the cache retains | unset |
| `max_concurrent_translations` | How many models can be decoding at once | unset |

`model_cache_size` alone is a poor bound because model sizes differ by more
than an order of magnitude: four Marian models are about 1.4 GB, four
MADLAD-400-3B are about 20 GB. `max_loaded_mb` puts a byte budget on the same
cache, evicting least-recently-used until the budget is met.

Neither reaches a model that is *in flight*. A model being decoded through is
resident because a thread is using it, not because the cache kept it, so
eviction cannot release it. `max_concurrent_translations` is the only key that
bounds this: requests over the limit wait for a slot rather than loading
another model. Translation endpoints are commonly served from a threadpool, so
without it peak memory scales with whatever that pool admits.

Set all three. A cache budget without a concurrency limit is not a memory
bound.

## Sizing the ceiling

Resident memory tracks roughly:

```
peak RSS  ~=  86 MB  +  1.84 x ( max_loaded_mb  +  concurrency x largest_model_mb )
```

- **86 MB** is the process floor with no model loaded.
- **1.84** is the ratio of resident bytes to declared weight size — an ONNX
  session holds more than the file: arena allocations, the tokenizer, and the
  decode-time KV cache.
- **`largest_model_mb`** is the largest model a route may select, which is what
  `max_model_mb` and `oversize_fallback` decide. With `oversize_fallback: true`
  it is not `max_model_mb`; it is the largest model the fallback may escalate
  to, which is bounded by `LINGUONNX_MAX_DOWNLOAD_MB` (default 8192).

Worked example, for a 12 GB container:

```json
{
  "language": {
    "translation_module": "ovos-translate-plugin-linguonnx",
    "ovos-translate-plugin-linguonnx": {
      "precision": "int8",
      "max_model_mb": 500,
      "oversize_fallback": true,
      "count_cached_as_free": false,
      "max_loaded_mb": 2048,
      "max_concurrent_translations": 2
    }
  }
}
```

The escalation ceiling on the default registry is a 4945 MB MADLAD checkpoint,
so the worst case is `86 + 1.84 x (2048 + 2 x 4945)` — about 22 GB, which does
not fit. Either lower `LINGUONNX_MAX_DOWNLOAD_MB` so the fallback cannot reach
the largest checkpoints, or keep `max_concurrent_translations` at `1` and
accept that two simultaneous Chuvash requests queue. Sizing against the
*typical* model rather than the escalation ceiling is what produces a server
that runs for weeks and then dies under an unusual language pair.

## Bound the container too

Application-level bounds are a budget, not an enforcement mechanism. Give the
container a hard limit so an overrun is a restart rather than an OOM kill that
takes the host's other services with it:

```yaml
services:
  linguonnx:
    mem_limit: 12g
```

Confirm it applied rather than trusting the file — `docker inspect` reports the
effective value in bytes:

```bash
docker inspect <container> --format '{{.HostConfig.Memory}}'
```

Set the limit above the sizing calculation, not equal to it. The calculation
covers translation; it does not cover a download in progress or the language
detector's own model.

## Prefetch before serving traffic

A cold cache is slow, not broken: the first request for an uncached model
blocks on a real download, and a 4.9 GB model takes about 349 seconds to fetch
and load. On a public server that is indistinguishable from a hang.

Prefetch the int8 registry before routing traffic to the instance:

```python
from linguonnx import load_translator
from linguonnx.model_manager import prefetch

prefetch(*load_translator(precision="int8").models, kind="translate")
```

Budget roughly 109 GB of disk for the full int8 registry, on a volume that
survives container replacement. Prefetching also makes routing deterministic
across a fleet: while the effective download budget is below the largest
runnable model, a cached model costs no download and the budget never sees it,
so a warm host and a cold host can pick different routes for the same request.

## Health checks

`/status` answers without loading a model, so it is the right endpoint for a
container health check or a load balancer. A translation request is not — it
can block on a cold download and will mark a healthy instance as failed.

```bash
curl http://localhost:9686/status
```

To verify routing behaviour after a config change, translate a pair that
exercises the branch you care about rather than a pair every config handles:

```bash
curl http://localhost:9686/translate/en/ca/hello%20world   # small dedicated model
curl http://localhost:9686/translate/en/cv/hello%20world   # oversize fallback
```

## Verify the configuration actually applied

The plugin reads `language.<plugin-id>` from `mycroft.conf`. A config placed
under any other section is silently ignored — no error, no warning, and the
server runs on library defaults while appearing configured. This is the single
most common deployment fault.

```json
{
  "language": {
    "translation_module": "ovos-translate-plugin-linguonnx",
    "detection_module": "ovos-lang-detect-plugin-linguonnx",
    "ovos-translate-plugin-linguonnx": {},
    "ovos-lang-detect-plugin-linguonnx": {}
  }
}
```

Prove it applied by asking the running instance to route a pair whose answer
differs between your config and the defaults. With the 12 GB example above,
`en -> ca` routes through a 157 MB `opus-mt` model; on defaults it routes
through a 1.7 GB multilingual model. Watching which one loads is the check —
reading the config file back is not, since the file being correct and the
plugin reading it are different claims.
