"""OVOS translation plugin wrapping the linguonnx offline translator."""

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
    ``max_loaded_mb``
        Byte budget for the loaded-model cache, in MB. Unset by default, so
        ``model_cache_size`` alone bounds it. Four Marian models are about
        1.4 GB and four MADLAD-400-3B are about 20 GB, so a count alone is a
        poor bound. This bounds what the cache *retains*; it does not bound
        peak memory - see ``max_concurrent_translations``.
    ``max_concurrent_translations``
        How many translations may run at once. Unset by default, meaning no
        limit. This is the only key that bounds peak memory: a model being
        translated through is resident because a thread is decoding with it,
        not because the cache kept it, so no cache setting can reach it. OVOS
        translation endpoints are commonly served from a threadpool, so peak
        memory otherwise scales with whatever that pool admits. Requests over
        the limit wait for a slot instead of loading another model.
    ``pivot_preference``
        Ordered pivot languages to try on a two-hop route, overriding
        linguonnx's default preference list.
    ``max_routes``
        How many candidate routes are scored before the best is taken.
    ``length_penalty``
        Beam-search length penalty. Above 1.0 favours longer output.
    ``no_repeat_ngram_size``
        Block repeating any n-gram of this size. ``0`` (default) disables it.
    ``max_model_mb``
        Drop any model bigger than this, in MB, from the routing graph before
        a route is even scored - a *routing* filter, not a download guard.
        Unset by default, which leaves every model in the graph eligible.
        Combining this with ``models`` makes linguonnx treat it as an
        operator-set budget that outranks the ``models`` waiver, per
        linguonnx's own ``operator_budget_is_set()`` rule; see the linguonnx
        docs before setting both.
    ``oversize_fallback``
        ``False`` by default, which makes ``max_model_mb`` a filter: a
        language that lives only inside an oversized model becomes
        unroutable. ``True`` makes the cap a *preference* instead - every
        pair a model under the cap can serve is still served by that model,
        and only a pair nothing under the cap covers escalates to the
        smallest oversized model that does. Set this on a host that caps for
        load latency rather than for disk, so the cap does not also delete
        the long tail of languages.
    ``count_cached_as_free``
        Whether a model already in the local cache is exempt from
        ``max_model_mb``. Left unset, linguonnx picks ``True`` normally and
        ``False`` when ``oversize_fallback`` is on, because a warm cache
        would otherwise exempt every model there is and make the cap a
        no-op. Set it explicitly only to overrule that: ``True`` reads the
        cap as "do not download more than this", ``False`` as "do not load a
        model bigger than this".
    ``exclude_flagged``
        ``False`` by default. Drop any model linguonnx's own quality sweep
        flags - either precision scoring below 40 chrF against the FLORES-200
        reference, or int8 trailing fp32 by more than 2 chrF. Combine with
        ``precision: null`` to fall back to fp32 wherever int8 alone is
        flagged.
    ``min_chrf``
        Drop any model scoring below this chrF against FLORES-200. Unset by
        default, which keeps every model the other filters allow through.
    ``models``
        Exact registry ids to route over, overriding every other filter. A
        list of model ids as linguonnx names them.
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
        passthrough = ("models", "model", "prefer", "max_hops", "pivot_ranking",
                       "pivot_preference", "max_routes",
                       "include_noncommercial", "precision", "exclude_flagged",
                       "min_chrf", "model_cache_size", "max_loaded_mb",
                       "max_concurrent_translations",
                       "num_beams", "max_new_tokens", "length_penalty",
                       "no_repeat_ngram_size",
                       "max_model_mb", "oversize_fallback", "count_cached_as_free")
        return {k: self.config[k] for k in passthrough if k in self.config}

    # -- engine -----------------------------------------------------------

    @property
    def translator(self):
        """Lazily build (and cache) the linguonnx Translator."""
        if self._translator is None:
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
