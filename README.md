# NLU Assignment 2 - Japneet Singh (B23CS1022)

This repository contains implementations and analysis for two NLP tasks:

1. **Problem 1: Learning Word Embeddings from IIT Jodhpur Data**
2. **Problem 2: Character-Level Name Generation Using RNN Variants**

The work is implemented primarily in notebooks:
- `prob1.ipynb` for Word2Vec corpus creation, training, semantic analysis, and visualization
- `prob2.ipynb` for character-level name generation models and evaluation

---

## Repository Structure

```text
.
|-- prob1.ipynb
|-- prob2.ipynb
|-- combined_report.tex
|-- iitj_raw_corpus.txt
|-- iitj_clean_corpus.txt
|-- TrainingNames.txt
|-- vanilla_rnn_state_dict.pt
|-- iitj_pdfs/
|-- report_assets/
|-- index.html
|-- README.md
```

---

## Problem 1: Word Embeddings from IITJ Data

### Objective
Build a domain corpus from IIT Jodhpur sources and train Word2Vec models (**CBOW** and **Skip-gram with Negative Sampling**) to analyze semantic structure.

### Implemented Pipeline (`prob1.ipynb`)
- Web scraping from IITJ academic/department/research/faculty/announcement pages
- PDF text extraction from `iitj_pdfs/`
- Corpus preprocessing:
  - boilerplate and URL cleanup
  - tokenization (NLTK)
  - lowercasing
  - stopword removal (+ custom IITJ/navigation stopwords)
  - lemmatization
- Saved corpora:
  - raw: `iitj_raw_corpus.txt`
  - clean: `iitj_clean_corpus.txt`
- Word cloud and token statistics
- From-scratch NumPy Word2Vec:
  - CBOW with negative sampling
  - Skip-gram with negative sampling
- Semantic analysis:
  - nearest neighbors by cosine similarity
  - analogy solving
- 2D visualization:
  - t-SNE
  - PCA

### Key Reported Stats
- Total documents: **71**
- Total tokens: **640,046**
- Vocabulary size: **34,446**

---

## Problem 2: Character-Level Name Generation (RNN Variants)

### Objective
Train and compare recurrent architectures for Indian name generation at character level.

### Implemented Models (`prob2.ipynb`)
1. **Vanilla RNN**
2. **BLSTM**
3. **RNN with Basic Additive Attention**

### Dataset
- `TrainingNames.txt` contains 1000 cleaned Indian names
- Includes sequence tokens: `SOS (<)`, `EOS (>)`, `PAD (#)`

### Evaluation Metrics
- **Novelty Rate**: generated names not present in training set
- **Diversity**: unique generated names / total generated names

### Reported Quantitative Results
- Vanilla RNN: Novelty **97.19%**, Diversity **99.40%**
- BLSTM: Novelty **100.00%**, Diversity **73.79%**
- Attention: Novelty **75.75%**, Diversity **95.39%**

---

## Setup and Run

### 1. Create and activate virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install jupyter notebook numpy scipy matplotlib scikit-learn nltk requests beautifulsoup4 PyPDF2 wordcloud gensim torch indian-names
```

### 3. Download NLTK resources (if not auto-downloaded)

The notebooks already call `nltk.download(...)`, but if needed:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Run notebooks

```powershell
jupyter notebook
```

Then open and run:
- `prob1.ipynb` cells in order (top to bottom)
- `prob2.ipynb` cells in order (top to bottom)

---

## Reproducibility Notes

- Random seeds are set in both notebooks (`SEED = 42`) for more stable outputs.
- `prob1.ipynb` depends on internet access for webpage scraping.
- `prob1.ipynb` also depends on local PDFs in `iitj_pdfs/`.
- `prob2.ipynb` can regenerate `TrainingNames.txt` using `indian-names` package.

---

## Deliverables Mapping

- **Source code**: `prob1.ipynb`, `prob2.ipynb`
- **Cleaned corpus file**: `iitj_clean_corpus.txt`
- **Visualizations**: notebook outputs and `report_assets/`
- **Report**: `combined_report.tex`
- **Saved model artifact (example)**: `vanilla_rnn_state_dict.pt`

---

## Author

- **Japneet Singh**
- **Roll No:** B23CS1022
