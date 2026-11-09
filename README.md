# Fake News Detector

[![CI](https://github.com/gorintlapavanprasad/Fake_news_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/gorintlapavanprasad/Fake_news_Detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://gorintlapavanprasad.github.io/Fake_news_Detection/)

**An open-source fake news classifier that runs entirely in your browser and shows which words drove each decision.**

**[Try the live demo](https://gorintlapavanprasad.github.io/Fake_news_Detection/)**. Paste an article, get REAL or FAKE, and see the evidence words marked in red and green inside your own text. Nothing is uploaded and no API keys are used.

Keywords: fake news detection, misinformation, NLP, text classification, TF-IDF, logistic regression, DistilBERT, explainable AI, GitHub Pages, JavaScript, Python, scikit-learn.

## What is different here

Most fake-news repos report 99% on one dataset and stop. This project measures what happens on other data, and says so.

| Test | Result |
|---|---|
| Held-out articles, ISOT (Reuters vs. fake-news sites) | 97.1% |
| Held-out articles, WELFake (mixed outlets, no overlap with ISOT) | 89.9% |
| Trained on ISOT alone, tested on WELFake | about 50% (chance) |
| Celebrity gossip (GossipCop), never seen in training | 41.7% (below chance) |

- The original dataset has giveaways, such as "(Reuters)" datelines, that let a model score about 99% without learning anything about truth. They are stripped before training.
- A model trained on one source collapses on another. Training on two sources fixed most of that for news, but not for a new genre such as celebrity gossip. The site shows this limit openly.
- DistilBERT was fine-tuned too (95.6% and 84.1% on 800 held-out articles per dataset, 3,000 training articles, one epoch, CPU only). It did not beat the TF-IDF model under that budget, and at about 270 MB it cannot ship in a static site.

## How it works

```
article text -> clean (strip links, digits, source giveaways)
             -> TF-IDF over 30,000 words and word pairs
             -> logistic regression -> probability of REAL
             -> per-word contributions -> highlighted evidence
```

The same arithmetic is implemented twice: in Python (`improved/`) for training and in JavaScript (`docs/engine.js`) for the website. `tests/test_parity.mjs` feeds both the same texts and requires agreement within 0.001 (measured worst gap: 8e-7). `tests/eval_js_engine.mjs` scores the JavaScript engine on 1,500 held-out articles per dataset and requires its accuracy to match Python's.

## Run it

Website (no build step):
```bash
cd docs && python -m http.server 8000     # open http://localhost:8000
```

Retrain from scratch:
```bash
pip install -r requirements-dev.txt
# put Fake.csv and True.csv (Kaggle "Fake and Real News", ISOT) in Prototype/data/raw/
cd improved
python run_experiments.py --data ../Prototype/data/raw      # 16 model/feature variations
python prepare_external.py --isot ../Prototype/data/raw     # second dataset (WELFake), see below
python export_web.py --isot ../Prototype/data/raw --ext external/welfake_clean.csv
```

Tests:
```bash
pytest tests
node tests/test_parity.mjs
node tests/test_engine_robustness.mjs
```

Data sources: ISOT from [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets); WELFake (`davanstrien/WELFake`) and GossipCop (`Jinyan1/GossipCop`) from Hugging Face. Check each dataset's license before redistributing it. The repo stores no datasets, only a 25-article sample (`docs/parity.json`) used to test the browser engine.

## Repository layout

| Path | What it holds |
|---|---|
| `docs/` | The static website served by GitHub Pages, its model file and the design notes |
| `improved/` | Cleaning code, experiments, external-data preparation, DistilBERT fine-tuning, Flask demo |
| `tests/` | Python unit tests and Node tests for the browser engine |
| `Prototype/` | Original PyTorch BiLSTM + attention model (config-driven) |
| `main.py`, `FND_using_*.py` | Original TensorFlow hybrid and classical scripts, now runnable locally |
| `PROJECT_REPORT.md` | Full analysis, 16-experiment table, problems found, roadmap |

## Limits

It scores writing style learned from mostly English, US-politics articles from 2016 and 2017. It cannot verify facts, and it is unreliable on satire, opinion, other languages and unfamiliar topics such as entertainment news. Use it as a prompt to look closer, never as a verdict.

## Roadmap

1. Add evidence retrieval (search or fact-check APIs) so claims, not only style, are checked. This needs a server and keys, so it cannot live on GitHub Pages.
2. Fine-tune DistilBERT on the full data with a GPU and compare on the unseen-source tests.
3. Collect more varied training sources (entertainment, health, science) and recalibrate the probabilities.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Released under the [MIT license](LICENSE). Cite via [CITATION.cff](CITATION.cff).

Authors: Pavan Gorintla, Smirthya Somaskantha Iyer, Rahul Rubugunday.
