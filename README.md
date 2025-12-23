# parl_speech_topic_sent
Analyzing how policy makers spoke on specific policy topicy in the Bundestag, leveraging NLP.

research question: “Wie unterscheiden sich Themen und Sentiment zwischen Fraktionen im Deutschen Bundestag über Zeit, und wie verändern sich diese Muster in politischen Krisen?”

Operationalisierung:
- Themen = Topics aus Topic Modeling auf speechContent
- Sentiment = Scores (−, 0, +) aus einem Sentiment-Modell
- Gruppen = factionId / Partei, positionShort (Minister, MP, Präsidium …)
- Zeit = date (z.B. nach Legislaturperiode / Jahr / vor/nach Event)
- Damit hast du eine klar politikwissenschaftliche Frage + spannenden NLP-Teil.

## Pipeline Execution Order

Execute the following scripts and notebooks in this exact order:

### 1. Data Preparation
1. **`scripts/convert_csv_parquet.ipynb`** (optional)
   - Converts `speeches.csv` to `speeches.parquet` for faster loading
   - Input: `data/raw/speeches.csv`
   - Output: `data/raw/speeches.parquet`

2. **`scripts/sample_data.py`**
   - Creates a random sample of 1000 speeches from the full dataset
   - Input: `data/raw/speeches.parquet` (or .csv)
   - Output: `data/raw/df_sample.csv`

### 2. Data Cleaning
3. **`notebooks/data_cleaning.ipynb`**
   - Cleans and preprocesses the sample data
   - Input: `data/raw/df_sample.csv`
   - Output: `data/processed/df_sample_cleaned.csv`

### 3. Text Segmentation & Preprocessing
4. **`notebooks/text_segmentation_tokenization_topic.ipynb`**
   - Splits speeches into paragraphs
   - Tokenization, lemmatization, and preprocessing for topic modeling
   - Input: `data/processed/df_sample_cleaned.csv`
   - Output: `data/interim/df_sample_split.csv`, `data/processed/df_sample_split_preprocessed_topic.parquet`

### 4. NLP Analysis (can be run in parallel)
5. **`notebooks/topic_modeling.ipynb`**
   - Performs LDA and BERTopic modeling
   - Input: `data/processed/df_sample_split_preprocessed_topic.parquet`
   - Output: `data/processed/topic_document_assignments_bert.parquet` (+ various topic model outputs)

6. **`notebooks/sentiment_analysis.ipynb`**
   - Analyzes sentiment using German sentiment model
   - Input: `data/interim/df_sample_split.csv`
   - Output: `data/processed/df_sample_sentiment.parquet`

### 5. Merge & Analysis
7. **`notebooks/merge_analysis_df.ipynb`**
   - Merges topic and sentiment data
   - Input: `data/processed/df_sample_sentiment.parquet`, `data/processed/topic_document_assignments_bert.parquet`
   - Output: `data/processed/df_final_analysis.parquet`

8. **`notebooks/final_analysis.ipynb`**
   - Final visualizations and statistical analysis
   - Input: `data/processed/df_final_analysis.parquet`
   - Output: Tables in `reports/tables/`, figures in `reports/figures/`