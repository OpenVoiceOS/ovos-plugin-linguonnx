"""OVOS translation plugin wrapping the linguonnx offline translator."""

import os
from typing import Dict, Optional, Set

from ovos_plugin_manager.templates.language import LanguageTranslator

#: Fallback target when neither the caller nor the config names one.
DEFAULT_LANG = "en"


class LinguONNXTranslatePlugin(LanguageTranslator):
    """LanguageTranslator implementation backed by ``linguonnx``.

    Translation runs entirely offline on ONNX Runtime. linguonnx does not hold
    one giant multilingual model; it holds a *graph* of models and routes a
    request through one or two of them, so a pair with no direct model
    (``pt->eu``) is served by pivoting (``pt->en->eu``).

    Configuration keys, all optional:

    ``prefer``
        Route ranking. ``"fewest_hops"`` (default) takes the shortest route;
        ``"dedicated"`` prefers a bilingual model over a multilingual one even
        when that costs an extra hop.
    ``max_hops``
        How many models a route may chain. ``1`` is direct models only,
        ``2`` (default) allows one pivot language. Higher is allowed but each
        hop compounds the previous hop's errors.
    ``pivot_ranking``
        How pivot candidates are ordered on a two-hop route. ``"auto"``
        (default) uses phonological distance when ``orthography2ipa`` is
        installed and a curated table otherwise; ``"phonological"`` demands the
        package; ``"table"`` ignores it.
    ``include_noncommercial``
        ``False`` by default, so only permissively licensed models are used.
        ``True`` adds NLLB-200 (CC-BY-NC-4.0) - broader coverage, but it puts a
        non-commercial licence on the output.
    ``precision``
        ``"int8"`` (default) for the quantized models, ``"fp32"`` for the full
        ones (roughly 4x the disk and memory), or ``null`` for both.
    ``model_cache_size``
        How many loaded models are kept in memory at once, least-recently-used
        evicted first. Defaults to linguonnx's own default (4). The whole
        default graph is ~25 GB, so this is the knob that keeps a long-lived
        server from being OOM-killed.
    ``max_model_mb``
        Refuse a cold download bigger than this, in MB, instead of holding the
        request for minutes. Unset by default, which leaves linguonnx's 8192 MB
        budget in place. Set it low on a host whose cache is pre-warmed, so a
        surprise fetch fails fast and loudly.
    ``num_beams``
        Beam width. ``4`` by default; ``1`` is greedy and about 4x faster.
    ``max_new_tokens``
        Output length cap per hop, ``128`` by default. Raise it for long
        input; the decode loop stops at the cap without warning.

    The translator is built on first use, so constructing this class never
    reads the disk or the network.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._translator = None
        self._targets: Dict[str, Set[str]] = {}

    # -- config -----------------------------------------------------------

    @property
    def default_language(self) -> str:
        """Target used when the caller gives none. ``internal`` beats ``lang``."""
        return self.config.get("internal") or self.config.get("lang") or DEFAULT_LANG

    @property
    def loader_kwargs(self) -> Dict:
        """Config keys passed straight through to ``linguonnx.load_translator``.

        Keys the config does not set are left out entirely rather than
        defaulted here, so linguonnx stays the single source of truth for what
        a default is.
        """
        passthrough = ("prefer", "max_hops", "pivot_ranking",
                       "include_noncommercial", "precision",
                       "model_cache_size", "num_beams", "max_new_tokens")
        return {k: self.config[k] for k in passthrough if k in self.config}

    # -- engine -----------------------------------------------------------

    @property
    def translator(self):
        """Lazily build (and cache) the linguonnx Translator."""
        if self._translator is None:
            # linguonnx reads the download budget from the environment at
            # fetch time, and there is no load_translator argument for it.
            max_model_mb = self.config.get("max_model_mb")
            if max_model_mb is not None:
                os.environ["LINGUONNX_MAX_DOWNLOAD_MB"] = str(int(max_model_mb))
            from linguonnx import load_translator
            self._translator = load_translator(**self.loader_kwargs)
        return self._translator

    # -- LanguageTranslator API -------------------------------------------

    def translate(self, text: str, target: Optional[str] = None,
                  source: Optional[str] = None) -> str:
        """Translate ``text`` from ``source`` to ``target``.

        Both tags may carry a region (``pt-PT``, ``en-US``); linguonnx
        normalizes them. Unset arguments fall back to the configured language.
        """
        from linguonnx.model_manager import DownloadTooLargeError
        from linguonnx.translate import NoRouteError

        target = target or self.default_language
        source = source or self.config.get("lang") or DEFAULT_LANG
        try:
            return self.translator.translate(text, src=source, tgt=target)
        except NoRouteError as err:
            # An unsupported pair is the caller's problem, not a fault. Callers
            # that only know the OVOS interface cannot import linguonnx's
            # exception types to tell the two apart, so this becomes a plain
            # ValueError and a server can answer 4xx instead of 5xx. linguonnx's
            # own message says which bound blocked the pair, so it is kept.
            raise ValueError(str(err)) from err
        except DownloadTooLargeError as err:
            # The pair is routable; the host simply has not cached the model and
            # the download budget refused to fetch it on the request path. That
            # is a deployment state, so it stays a RuntimeError - retrying the
            # same request will not help until someone prefetches.
            raise RuntimeError(str(err)) from err

    @property
    def available_languages(self) -> Set[str]:
        """Every tag reachable as a source or a target across the whole graph."""
        return set(self.translator.available_languages)

    def supported_translations(self, source_lang: str) -> Set[str]:
        """Targets actually routable from ``source_lang``.

        This is narrower than :attr:`available_languages`, which is the union
        over the graph: a tag can be in that union and still have no route from
        a given source. Answers are cached per source, because the graph does
        not change once the translator is built.
        """
        if source_lang not in self._targets:
            tx = self.translator
            self._targets[source_lang] = {
                lang for lang in tx.available_languages
                if lang != source_lang and tx.can_translate(source_lang, lang)}
        return self._targets[source_lang]


__all__ = ["LinguONNXTranslatePlugin", "DEFAULT_LANG"]
