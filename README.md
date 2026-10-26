# Customer Feedback Sentiment Analysis and Analytics Dashboard

A beginner-friendly portfolio project combining Python, NLP, statistics, probability, data visualization, Agile planning, and stakeholder engagement.

## Features
- Predicts positive, neutral, and negative sentiment from customer feedback.
- Uses TF-IDF text features and Logistic Regression.
- Displays sentiment counts, percentages, charts, and confidence summaries.
- Allows CSV uploads, text search, sentiment filters, and CSV export.
- Includes a small sample dataset for demonstration.

## Important model limitation
The bundled model is trained on a tiny, hand-written demonstration dataset. It is a learning prototype, not a production-ready sentiment model. Results may be inaccurate for sarcasm, domain-specific wording, mixed sentiment, or languages other than English. For credible evaluation, collect a representative labelled dataset, create a separate train/test split, compare baselines, and report precision, recall, F1-score, and confusion matrix.

## Live demo
[Open the Streamlit dashboard](https://customer-feedback-sentiment-zmtwbo68na4vdwjf3viqbd.streamlit.app/)

## Requirements
- Python 3.10 or newer recommended
- pip

## Run locally
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in the terminal. Upload `sample_feedback.csv` from the sidebar to test CSV upload.

## CSV format
Your CSV must include a `text` column. A `sentiment` column is optional and should contain `positive`, `neutral`, or `negative` if you want to compare predictions with labels.

Example:
```csv
text,sentiment
"The service was helpful",positive
"Delivery arrived late",negative
"The product has two colours",neutral
```

## Project workflow
1. Define the business question: understand customer experience from text feedback.
2. Gather and validate feedback data, remove empty rows, and review privacy considerations.
3. Preprocess text and convert it into TF-IDF features.
4. Train a Logistic Regression baseline.
5. Analyze sentiment distribution and confidence summaries.
6. Present insights in a Streamlit dashboard.
7. Review findings with stakeholders and iterate based on feedback.

## Agile sprint plan
| Sprint | Focus | Deliverable |
|---|---|---|
| Sprint 1 | Requirements, sample data, data quality checks | User stories and validated CSV format |
| Sprint 2 | NLP baseline and sentiment predictions | Working classification pipeline |
| Sprint 3 | Statistics and dashboard visualizations | Charts, summary metrics, filters |
| Sprint 4 | Testing, stakeholder review, documentation | Tested prototype and improvement backlog |

## Example user stories
- As a customer experience analyst, I want to see sentiment distribution so I can identify patterns in feedback.
- As a product manager, I want to filter negative feedback so I can review issues requiring attention.
- As a stakeholder, I want an exportable results file so I can share findings with my team.

## Stakeholder engagement
For a portfolio demonstration, document the stakeholder role, requirement, feedback, decision, and resulting change. Do not claim real stakeholder interviews or team collaboration unless you actually performed them.

## Suggested improvements
- Train and evaluate on a larger public or approved dataset.
- Add confusion matrix and per-class precision, recall, and F1-score.
- Add topic extraction to identify common complaint themes.
- Add date and product-category trend analysis if those fields are available.
- Add tests and a deployment workflow.

## Resume bullet (use after you run and verify the project)
Built a Python-based customer feedback analytics dashboard using NLP and machine learning to classify sentiment, summarize feedback patterns with statistical metrics, and visualize insights in Streamlit; organized development into Agile-style sprints and documented stakeholder-oriented requirements.
