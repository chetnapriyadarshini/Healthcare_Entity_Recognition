# Healthcare Entity Recognition — Named Entity Recognition with CRF

A Jupyter Notebook implementing a **Conditional Random Field (CRF)** model for Named Entity Recognition (NER) on a healthcare dataset of patient-doctor interaction logs, automatically extracting disease entities and their associated treatment recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Notebook Contents](#notebook-contents)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Results](#results)
- [References](#references)
- [Contact](#contact)

---

## Overview

Medical records and clinical interaction logs contain rich, unstructured information about diagnoses and treatments. Manually extracting this information at scale is time-consuming and error-prone. This project builds an NER pipeline that automatically identifies **disease** and **treatment** entities in patient-doctor dialogue, constructing a structured disease-to-treatment dictionary from unstructured clinical text.

The practical application is significant: such a system can assist in clinical decision support, pharmacovigilance, and medical knowledge base construction.

---

## Background

### Named Entity Recognition (NER)

NER is a sequence labelling task in NLP — each token in a text is assigned a label from a predefined entity schema. In a medical context, a token might be labelled as `B-Disease` (beginning of a disease entity), `I-Disease` (continuation), `B-Treatment`, `I-Treatment`, or `O` (outside any entity). This is known as **BIO tagging**.

### Why CRF?

Conditional Random Fields model the conditional probability of the entire label sequence given the input sequence, capturing dependencies between adjacent labels. This is important in NER because entity labels are not independent — a token labelled `I-Disease` should only follow `B-Disease` or another `I-Disease`, never `O`. CRFs explicitly model these transition constraints, making them well-suited to structured sequence labelling tasks.

```
Input:  ["The", "patient", "has", "diabetes", "and", "was", "prescribed", "metformin"]
Labels: ["O",   "O",       "O",  "B-Disease", "O",  "O",  "O",          "B-Treatment"]
```

---

## Dataset

The dataset consists of patient-doctor interaction logs in which clinical dialogues describe symptoms, diagnoses, and prescribed treatments. Each sentence is tokenised and annotated at the token level with entity labels. The corpus covers a range of medical conditions and their standard treatment protocols.

---

## Methodology

The project follows a four-stage pipeline:

**1. Feature Engineering**

For each token in a sentence, a feature vector is constructed capturing:
- Word identity (current token, previous token, next token)
- Word shape features (is it capitalised, all-uppercase, contains digits)
- Part-of-speech tag
- Prefix and suffix characters
- Position features (beginning of sentence, end of sentence)
- Medical vocabulary indicators

**2. Feature Application**

The defined feature functions are applied across every token in both the training and test datasets, producing feature-value mappings for each token in context.

**3. CRF Model Training**

A CRF model is trained on the feature-annotated training corpus, learning both the emission probabilities (which features associate with which entity types) and the transition probabilities (valid label sequences).

**4. Disease-Treatment Dictionary Construction**

Predicted entity spans are extracted from the test set and mapped into a structured Python dictionary where disease entities are keys and their co-occurring treatment entities are values — creating a machine-readable medical knowledge resource.

---

## Notebook Contents

| Section | Description |
|---|---|
| Data Loading & Inspection | Loading patient-doctor interaction logs, inspecting sentence and token distributions |
| BIO Tag Analysis | Exploring entity label distribution across the corpus |
| Feature Engineering | Defining token-level features (word shape, POS, context window, affixes) |
| Feature Application | Applying feature functions across all tokens in train and test sets |
| CRF Model Training | Training the CRF model using `sklearn-crfsuite` |
| Evaluation | Precision, Recall, F1-score per entity type (Disease, Treatment) |
| Disease-Treatment Dictionary | Extracting entity pairs into a structured disease → treatment mapping |
| Error Analysis | Reviewing misclassified entities and common failure modes |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `sklearn-crfsuite` | CRF model training and evaluation |
| `spaCy` / `nltk` | Tokenisation and part-of-speech tagging |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Label distribution visualisation |

---

## Setup and Installation

```bash
git clone https://github.com/chetnapriyadarshini/Healthcare_Entity_Recognition.git
cd Healthcare_Entity_Recognition
pip install sklearn-crfsuite spacy nltk pandas numpy matplotlib seaborn
python -m spacy download en_core_web_sm
jupyter notebook Chetna_Priyadarshini.ipynb
```

---

## Results

The CRF model successfully identifies disease and treatment entities in unseen clinical interaction logs. Evaluation metrics (Precision, Recall, F1) are reported per entity type in the notebook. The final output is a structured disease-to-treatment dictionary extracted from the test corpus, demonstrating the system's practical utility for automated medical knowledge extraction.

---

## References

- Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*. ICML 2001.
- Uzuner, Ö. et al. (2011). *2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text*. JAMIA.

---

## Contact

Created by [@chetnapriyadarshini](https://github.com/chetnapriyadarshini) — feel free to reach out with questions or suggestions.
