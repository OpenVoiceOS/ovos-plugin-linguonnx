# Docker

This repo builds and publishes `ghcr.io/openvoiceos/ovos-plugin-linguonnx`, an
image that runs both plugins behind
[`ovos-translate-server`](https://github.com/OpenVoiceOS/ovos-translate-server)
on port `9686` -- `opm.lang.translate` at `--tx-engine` and `opm.lang.detect`
at `--detect-engine`.

## Quick start

```bash
docker run -p 9686:9686 -v linguonnx-cache:/home/ovos/.cache \
    ghcr.io/openvoiceos/ovos-plugin-linguonnx:dev
```

```bash
curl http://localhost:9686/status
curl http://localhost:9686/translate/en/gl/hello%20world
curl http://localhost:9686/detect/bom%20dia
```

## What's in the image

| | |
|---|---|
| Base | `python:3.12-slim` |
| Python | this plugin (local checkout) + `linguonnx[distance,opennmt,indic]` (pinned commit) + `ovos-translate-server` (pinned branch) |
| Port | `9686` |
| User | non-root, uid `1000` (`ovos`) |

## Model cache -- mount this

Models are **not** baked into the image; the full int8 registry is roughly
109 GB. They download from HuggingFace into the cache on first use and are
kept there across restarts. Mount a persistent volume over the whole cache
directory, not just one subdirectory:

```yaml
volumes:
  - linguonnx-cache:/home/ovos/.cache
```

This covers both `~/.cache/huggingface` (raw HF blobs) and
`~/.cache/linguonnx` (linguonnx's own model store). Both are created and
`chown`ed to the `ovos` user at build time, so a bind mount over an empty host
directory does not leave them root-owned and unwritable.

**A cold cache is slow, not broken.** The first request for an uncached model
blocks on a real download; a 4.9 GB model takes about 349 seconds to fetch and
load. Set `max_model_mb` (see [configuration.md](configuration.md)) if you
would rather a cold request fail fast than hang, and prefetch before routing
real traffic if that latency is unacceptable:

```python
from linguonnx import load_translator
from linguonnx.model_manager import prefetch

prefetch(*load_translator().models, kind="translate")
```

## Configuration

Both plugins read their config from `mycroft.conf`. Mount one in:

```yaml
volumes:
  - ./mycroft.conf:/home/ovos/.config/mycroft/mycroft.conf:ro
```

using the same `language_detection` / `language_translation` keys documented
in [language-detection.md](language-detection.md) and
[configuration.md](configuration.md). `OMP_NUM_THREADS` (default `8` in the
image) and `HF_HOME` (`/home/ovos/.cache/huggingface`) are set as environment
variables and can be overridden with `docker run -e`.

## Building locally

```bash
docker build -t ovos-plugin-linguonnx .
```

The [`docker` workflow](../.github/workflows/docker.yml) builds on every PR
(build-only, no push) and publishes to
`ghcr.io/openvoiceos/ovos-plugin-linguonnx` on pushes to `master` (`latest`),
`dev` (`dev`), and version tags.
