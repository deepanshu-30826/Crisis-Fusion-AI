# AI-03 Crisis Report Fusion & Priority Ranking

An end-to-end multi-task machine learning system designed to ingest high-velocity crisis tweets/reports, perform semantic deduplication and clustering, classify actionable information categories, rank priority urgency for emergency triage, and retrieve matching source evidence via vector search.

## Project Structure
```
crisis-fusion/
├── data/
│   ├── raw/                # Gzipped raw tweets (*.json.gz) and TREC-IS labels.json
│   ├── processed/          # Cleaned & joined filtered_reports.csv, variants.csv
│   └── embeddings/         # Normalized dense vector embeddings (embeddings.npy)
├── src/
│   ├── data_loader.py      # Schema inspection, tweets/labels loading & join
│   ├── preprocessing.py    # Meaning-preserving text normalization (keeps negations)
│   ├── variant_generator.py# Deterministic robustness test variants (4 transform families)
│   ├── embeddings.py       # Sentence-transformers embedding & L2 normalization
│   ├── clustering.py       # HDBSCAN & Agglomerative clustering + Pairwise F1 metric
│   ├── classifier.py       # Logistic Regression & LinearSVC category classifiers
│   ├── priority.py         # Continuous urgency ranking model + NDCG evaluation
│   ├── retrieval.py        # FAISS inner product index for top-k evidence lookup
│   ├── inference.py        # Unified prediction pipeline and batch runner
│   └── evaluation.py       # Multi-component evaluation scoring 100-pt weighted benchmark
├── models/                 # Saved classifier, priority, and FAISS index artifacts
├── scripts/
│   ├── prepare_data.py     # End-to-end data ingestion & embedding pipeline
│   ├── train.py            # Model training & parameter tuning
│   ├── build_index.py      # Retrieval index builder
│   ├── predict.py          # Batch inference CLI
│   └── generate_mock_data.py # Realistic synthetic crisis dataset generator
├── outputs/                # Predictions and evaluation reports
├── notebooks/              # Exploratory data analysis & experiments
├── app.py                  # Streamlit crisis report triage interface
├── config.yaml             # Centralized configuration
└── requirements.txt        # Python package dependencies
```

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare data (or generate synthetic data for testing):
   ```bash
   python scripts/generate_mock_data.py
   python scripts/prepare_data.py
   ```
3. Train models and build retrieval index:
   ```bash
   python scripts/train.py
   python scripts/build_index.py
   ```
4. Run inference or launch Streamlit demo:
   ```bash
   python scripts/predict.py
   streamlit run app.py
   ```
