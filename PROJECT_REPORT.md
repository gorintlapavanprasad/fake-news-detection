# Fake News Detection: Project Report

## 1. What the project does
Classifies news articles as REAL (label 1) or FAKE (label 0) using the Kaggle
"Fake and Real News" (ISOT) dataset: `Fake.csv` and `True.csv`.

| Part | What it is |
|---|---|
| `main.py` | TensorFlow hybrid: Word2Vec + LDA topics + CNN + BiLSTM + attention |
| `FND_using_random_forest.py`, `FND_using_decision_tree.py` | Classical models on avg-Word2Vec + 10 LDA topic features |
| `Prototype/` | PyTorch BiLSTM + attention, config-driven (`config.yaml`) |
| `improved/` (new) | Fast TF-IDF pipeline, 16-variation experiment runner, CLI predictor, web app |

## 2. How to run
```
pip install -r requirements.txt            # plus: pip install flask
# put Fake.csv and True.csv in Prototype/data/raw/
cd improved
python run_experiments.py --data ../Prototype/data/raw --new-data new_data_sample.csv
python app.py                              # open http://127.0.0.1:5000
python predict.py "some article text"      # CLI
```
`run_experiments.py` also accepts `--new-data your.csv` (columns `text`, optional `title`, `label`; 1 = real, 0 = fake).

## 3. Problems found
### Fixed
1. Two scripts had Colab `!pip` lines (SyntaxError) and hardcoded `/content/...` paths. They now use `FND_DATA_DIR` (default `Prototype/data/raw`).
2. `main.py` had inverted labels: it printed FAKE when P(real) >= 0.5, and LIME class names were reversed. Fixed.
3. `requirements.txt` lacked torch, pyyaml, tqdm, joblib. Added.
4. No `.gitignore`; `__pycache__` and a venv remnant were committed. `.gitignore` added (the already-tracked files still need `git rm --cached`).
5. About 6,200 duplicate or empty articles (44,898 down to 38,658) sat on both sides of random splits. The new pipeline de-duplicates first (38,658 unique articles remain).

### Not fixed (need decisions or more work)
6. **Source leakage.** Real articles begin "CITY (Reuters) -" and fake ones carry style markers; models can score about 99% from these alone. The new pipeline strips them (see results), but subtler cues remain, for example attribution words such as "said" and "tuesday".
7. **`main.py` data leakage.** Tokenizer, LDA and Word2Vec are fit on all data before the train/test split, and the test set doubles as validation. The README says otherwise.
8. **Dataset limits.** One source pair, 2016-2017, US politics. The model learns the *style of two outlets*, not truth. It will not generalise to other topics or outlets, and it cannot verify facts.
9. **Date columns are unusable.** About 70% of dates are unparseable (almost all in `Fake.csv`), and real/fake cover different date ranges, so time-based splits fail. I used held-out news sections instead.
10. **Prototype (PyTorch) not evaluated.** Training takes about 65 min per epoch on CPU here (about 8 s per batch); I stopped it. Its committed 99.85% test score has the same leakage problem. `config.yaml` defaults to `cuda`, and `torch.load` is used without `weights_only`.
11. **Hybrid `main.py` and the RF/DT scripts have not been run end to end** (they need tensorflow, which is not installed, and long LDA training). Syntax is valid.
12. **Tiny "new data" test.** `new_data_sample.csv` has only 12 hand-written items, so it is a smoke test, not evidence. One item is misclassified by nearly every TF-IDF model.
13. Short text (about 1 word) was scored FAKE with 82% confidence. The web app now returns UNCLEAR under 10 words or below 70% confidence.

## 4. Results (16 variations, `improved/results/experiments.csv`)
"random" = stratified 80/20 split. "unseen sections" = trained on some news sections, tested on held-out ones (left-news, US_News, Middle-east, worldnews). 11,447 test articles.

| Experiment | Random acc | Unseen-section acc |
|---|---|---|
| LogReg, leaks kept | 99.13% | 97.38% |
| LinearSVC, leaks kept | **99.59%** | **98.89%** |
| LogReg, leaks stripped | 98.27% | 94.76% |
| LinearSVC, leaks stripped | 98.56% | 95.78% |
| LogReg, 5k vocab, stripped (**deployed**) | 98.33% | 95.80% |
| ComplementNB, stripped | 93.78% | 91.70% |
| Random Forest (TF-IDF), stripped | 96.25% | 86.34% |
| Decision Tree, stripped | 90.29% | 82.65% |
| LogReg, title only | 94.83% | 82.32% |
| LogReg, first 50 words only | 98.12% | 94.09% |

Takeaways:
- Linear models on TF-IDF beat the tree models, and are fast (about 30 s each) and explainable.
- Stripping source giveaways costs about 1 point on random splits and 2-4 points on unseen sections. That gap is the "easy" score the original reported, so the honest figure is **about 95-96% on unseen sections**.
- Headlines alone are weak (about 82% on unseen sections); the body text carries the signal.
- Original README claims (DT 94%, RF 97%, hybrid 99.5%) were not reproduced here.

## 5. The web app (`improved/app.py`)
Paste a headline or article; get REAL / FAKE / UNCLEAR, a confidence bar, the words pushing toward each class, and honest warnings (short text, low confidence, "style not facts"). JSON API: `POST /api/check`, `GET /api/health`. Verified in a browser with real-style and fake-style examples.

## 6. Roadmap
1. Add more diverse data (e.g. LIAR, FakeNewsNet, or WELFake) and test on it. This is the single biggest improvement.
2. Fine-tune DistilBERT or similar; compare against the TF-IDF baseline on the unseen-section and cross-dataset tests.
3. Fact-grounding: retrieve evidence (search or fact-check APIs) so the tool checks claims, not just style.
4. Calibrate probabilities and add an "out of domain" detector.
5. Run the Prototype PyTorch model on a GPU; add unit tests and CI; remove tracked caches.
6. Deploy (container + gunicorn), add rate limiting and logging.

## 7. Update: second dataset, DistilBERT and the website

New data: WELFake (24,418 articles after removing duplicates and any overlap with ISOT; its labels are flipped relative to its documentation, which was verified by reading samples) and GossipCop (entertainment news, used only for testing).

| Training data | Test on ISOT | Test on WELFake |
|---|---|---|
| ISOT only | 98.7% | 49.7% |
| WELFake only | 75.0% | 92.7% |
| Both | 97.1% | 89.9% |

Unseen source: the final model scores 41.7% on human-written GossipCop articles and 53.6% on machine-written ones. It tends to label real entertainment stories as fake. A style classifier trained on political news does not transfer to a different genre.

DistilBERT (3,000 training articles, 1 epoch, max length 128, CPU): 95.6% on ISOT and 84.1% on WELFake (800 held-out articles each). It trained in 38 minutes. The TF-IDF logistic regression trains in about a minute on all 62,000 articles and scores higher, so it ships in the website. A GPU run with the full data would be the fair comparison.

Website: `docs/` runs the model in the browser. Parity with Python is tested to 1e-3 and measured at 8e-7. A robustness test found and fixed a bug where words such as "constructor" produced NaN.

## 8. Update: five sources, leave-one-source-out, and what did not work

This section supersedes the GossipCop figures in section 7 (that model had not trained on GossipCop; the shipped model now has).

Sources: ISOT, WELFake, GossipCop (human-written), PolitiFact (human-written, 277 articles) and COVID-19 claims and tweets. Machine-written variants are excluded on purpose.

| Source | Held-out articles (seen source) | Source left out entirely |
|---|---|---|
| ISOT | 94.3% | 64.8% |
| WELFake | 86.9% | 47.4% |
| GossipCop | 76.2% | 44.0% |
| PolitiFact | 69.6% | 66.1% |
| COVID-19 | 86.2% | 46.1% |

Training recipe for the right-hand column (`loso_eval.py`): up to 12,000 articles per training source, balanced class weights, 30,000 TF-IDF features, logistic regression with C = 30. Recipe A (ISOT and WELFake only) gave similar numbers, so more sources did not improve transfer.

Negative results, kept as documented experiments:
- Model confidence, share of known words and text length do not predict errors on an unseen source (AUROC 0.44 to 0.64).
- Centroid similarity as an out-of-domain detector (`ood_eval.py`, `results/ood.csv`): AUROC 0.34 to 0.88 depending on the held-out source; below 0.5 for ISOT, GossipCop and PolitiFact. Not shipped.

Consequences for the website: it now lists both result columns, says the model cannot tell when it is outside its experience, and labels its output "Model score" rather than "confident".

Open problems: a reliable out-of-domain detector, a larger evaluation set for PolitiFact, and a GPU run of DistilBERT on the full data.
