/*
 * In-browser inference engine for the Fake News Checker.
 * Mirrors improved/fnd.py (cleaning) + scikit-learn TfidfVectorizer
 * (word 1-2 grams, sublinear tf, smooth idf, L2 norm) + LogisticRegression.
 * Parity with the Python model is enforced by tests/test_parity.mjs.
 */
(function (root) {
  "use strict";

  var LEAK_WORDS = new Set([
    "reuters", "getty", "images", "image", "via", "pic", "twitter", "com",
    "featured", "video", "watch", "photo", "screenshot", "wire", "century",
    "21st", "breaking", "read", "more", "source"
  ]);
  var DATELINE = /^.{0,120}?\(reuters\)\s*[-\u2013\u2014]?\s*/i;
  var OUTLET_TAG = /\s[-|]\s*(the )?(new york times|breitbart|washington post)\b.*$/i;
  var URL_RE = /http\S+|www\.\S+/g;
  var NON_ALPHA = /[^a-z\s]/g;

  function clean(text) {
    text = String(text || "");
    text = text.replace(DATELINE, "").replace(OUTLET_TAG, "");
    text = text.toLowerCase().replace(URL_RE, " ").replace(NON_ALPHA, " ");
    return text.split(/\s+/).filter(function (w) { return w && !LEAK_WORDS.has(w); });
  }

  /** Mirror of sklearn's word analyzer: tokens of >= 2 chars, then unigrams + bigrams. */
  function ngrams(words) {
    var toks = words.filter(function (w) { return w.length >= 2; });
    var out = toks.slice();
    for (var i = 0; i < toks.length - 1; i++) out.push(toks[i] + " " + toks[i + 1]);
    return out;
  }

  /**
   * @param {{vocab:Object, intercept:number}} model
   * @returns {{pReal:number, words:number, known:number, real:Array, fake:Array}}
   */
  function analyse(model, title, text) {
    var words = clean((title || "") + " " + (text || ""));
    var counts = Object.create(null), key;
    ngrams(words).forEach(function (g) { counts[g] = (counts[g] || 0) + 1; });

    var feats = [], norm = 0;
    for (key in counts) {
      // own-property check: words like "constructor" must not hit Object.prototype
      if (!Object.prototype.hasOwnProperty.call(model.vocab, key)) continue;
      var e = model.vocab[key];
      var v = (1 + Math.log(counts[key])) * e[0];
      feats.push([key, v, e[1]]);
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    var score = model.intercept, contrib = [];
    feats.forEach(function (f) {
      var c = (f[1] / norm) * f[2];
      score += c;
      contrib.push([f[0], c]);
    });
    contrib.sort(function (a, b) { return a[1] - b[1]; });
    // single-word contributions, used by the UI to annotate the original text
    var unigrams = Object.create(null);
    contrib.forEach(function (c) { if (c[0].indexOf(" ") < 0) unigrams[c[0]] = c[1]; });
    return {
      unigrams: unigrams,
      pReal: 1 / (1 + Math.exp(-score)),
      words: words.length,
      known: feats.length,
      fake: contrib.filter(function (c) { return c[1] < 0; }).slice(0, 8),
      real: contrib.filter(function (c) { return c[1] > 0; }).reverse().slice(0, 8)
    };
  }

  root.FND = { clean: clean, analyse: analyse };
})(typeof window !== "undefined" ? window : globalThis);
