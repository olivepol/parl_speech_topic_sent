# Parliamentary Speech Topic & Sentiment Analysis

Analyzing how policymakers speak on specific policy topics in the German Bundestag using NLP.

## Research Question

**Do SPD and CDU express systematically different sentiment patterns when discussing the same political topics between 2000 and 2021?**

### Operationalization

| Variable | Measurement |
|----------|-------------|
| Topics | LDA topic modeling on speechContent (15 topics -> 8 after filtering) |
| Sentiment | Categorical scores (negative=-1, neutral=0, positive=+1) from German BERT |
| Groups | Political parties via factionId (CDU=4, SPD=23) |
| Time | Year (2000-2021), aggregated from date column |

## Project Structure

```
parl_speech_topic_sent/
 config/model_params.py      # Centralized model parameters
 data/
    raw/                    # Original data
    interim/                # Intermediate data
    processed/              # Analysis-ready data
 notebooks/                  # Jupyter notebooks (01-06)
 reports/
    figures/                # Visualizations (PNG)
    paper/                  # LaTeX research note
    tables/                 # Statistics (CSV)
 scripts/                    # Python scripts
 src/                        # Reusable modules
 requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
huggingface-cli login
python scripts/import_data.py
```

Then run notebooks 01-06 in order.

## Pipeline

| Step | File | Description |
|------|------|-------------|
| 0 | scripts/import_data.py | Download from HuggingFace |
| 1 | scripts/sample_data.py | 50% stratified sample (CDU/SPD, 2000+) |
| 2 | 01_data_cleaning.ipynb | Clean text |
| 3 | 02_text_segmentation_tokenization_topic.ipynb | Tokenize |
| 4 | 03_topic_modeling.ipynb | LDA (15 topics) |
| 5 | 04_sentiment_analysis.ipynb | German BERT sentiment |
| 6 | 05_merge_analysis_df.ipynb | Merge data |
| 7 | 06_final_analysis.ipynb | Visualizations |

## Key Results

**8 Policy Topics:** Economy, Social Policy, Education, Budget, Europe/Climate, Foreign/Security, Public Admin, Legislation

**Findings:**
- Both parties: predominantly neutral sentiment (mean ~ -0.04)
- CDU more positive: Economy, Legislation, Budget
- SPD more positive: Education, Social Policy
- Most differences not statistically significant

## Data Sources

- [German Parliament Speeches](https://huggingface.co/datasets/emilpartow/german-parliament-speeches)
- [germansentiment](https://github.com/oliverguhr/german-sentiment-lib)
