# ovos-plugin-linguonnx served via ovos-translate-server.
#
# This plugin ships both opm.lang.detect (GlotLID) and opm.lang.translate
# (linguonnx routing graph). Neither entry point does anything on its own --
# they need a server front-end, so this image runs ovos-translate-server with
# both engines wired to this plugin, matching the production deployment at
# ovos-translate-servers/linguonnxsrv on ser9.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 ovos

WORKDIR /app
COPY . /app

# linguonnx and ovos-translate-server are pinned to explicit commits rather
# than floating on @dev, same as the production image, so a CI rebuild does
# not silently pick up an unrelated breaking change upstream. Bump these
# deliberately; do not switch to @dev.
#
# ovos-translate-server comes from a branch carrying two unreleased fixes:
# PR #51 makes it read mycroft.conf at all (without it plugin config below
# would do nothing), and PR #53 (stacked on #51) turns a plugin failure into
# a 400/503 with a reason instead of a bare 500.
#
# The linguonnx extras are not optional in practice:
#   distance -> orthography2ipa, what makes pivot_ranking="auto" rank pivots
#               by phonological distance instead of falling back to a table;
#   opennmt  -> sacremoses + subword-nmt, without which the Proxecto Nos
#               models are in the routing graph but raise ImportError the
#               moment a route picks one;
#   indic    -> IndicTrans2 preprocessing, same story for the Indic models.
#
# The plugin itself (this repo, opm.lang.detect + opm.lang.translate) is
# installed from the local checkout, so its version always matches the image.
# Both dependencies come from PyPI rather than git. A git install pins the
# image to a branch or a commit that no index knows about: it cannot be
# reproduced from the published metadata, it silently drifts when the branch
# moves, and it keeps building long after the branch is merged and deleted.
#
#   ovos-translate-server[mcp]>=0.9.1a1
#       The branch this used to track (fix/plugin-errors-are-not-500) is
#       merged; 0.9.1a1 is the first alpha carrying it, verified from the
#       wheel: plugin config resolves from mycroft.conf, and plugin failures
#       answer with a reason instead of a bare 500. The [mcp] extra pulls
#       fastmcp so the image can serve the MCP endpoint.
#
#   linguonnx[distance,opennmt,indic]>=0.12.0a1
#       The SHA this used to pin resolves to 0.10.0a1, but that pin no longer
#       holds: this plugin's own pyproject requires linguonnx>=0.12.0a1, and
#       the `pip install .` on the next line upgrades straight past the pin.
#       The floor is stated here to match, so the resolver is not silently
#       overruling the Dockerfile. The extras are why the line exists at all
#       -- the plugin depends on bare `linguonnx`, and without distance /
#       opennmt / indic those model families are in the routing graph but
#       raise ImportError the moment a route picks one.
RUN pip install --no-cache-dir \
        "ovos-translate-server[mcp]>=0.9.1a1" \
        "linguonnx[distance,opennmt,indic]>=0.12.0a1" \
    && pip install --no-cache-dir .

# Create the cache parents OWNED BY ovos before any bind mount lands on them.
# Docker creates missing mount parents as root, so if .cache is absent from
# the image the container gets a root-owned /home/ovos/.cache and the
# unprivileged server cannot create ~/.cache/huggingface or
# ~/.cache/linguonnx inside it -- that is a live PermissionError / 500, not a
# hypothetical (see linguonnxsrv incident, ser9).
RUN mkdir -p /home/ovos/.cache/huggingface /home/ovos/.cache/linguonnx \
    && chown -R ovos:ovos /home/ovos/.cache

USER ovos
ENV HOME=/home/ovos
# Named explicitly rather than left to HOME resolution, so a cold fetch has a
# writable target and degrades to a slow download instead of a 500.
ENV HF_HOME=/home/ovos/.cache/huggingface
# Bound so onnxruntime does not grab every core on a shared host. Tune to the
# deployment; unset to let onnxruntime pick automatically.
ENV OMP_NUM_THREADS=8

EXPOSE 9686

# Models are NOT baked into this image -- the full int8 set is ~109 GB. They
# are fetched into the mounted /home/ovos/.cache volume on first use and kept
# there across restarts. A cold model load is slow, not broken: a 4.9 GB
# model took ~349s to fetch and load in production. Prefetch the cache before
# traffic if that latency is unacceptable on first request; see README.md.
ENTRYPOINT ["ovos-translate-server", "--tx-engine", "ovos-translate-plugin-linguonnx", \
            "--detect-engine", "ovos-lang-detect-plugin-linguonnx", \
            "--port", "9686", "--host", "0.0.0.0"]
