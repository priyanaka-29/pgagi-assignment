# 🤖 AI Text Analyzer — PGAGI Assignment

**Author:** Podugu Priyanka  
**Role Applied:** Backend Development Intern – AI/ML  
**Company:** PGAGI Consultancy Pvt. Ltd  

---

## 📌 Project Overview

A multi-task NLP pipeline built using **Python** and **HuggingFace Transformers** that performs three AI tasks on any input text:

| Task | Model Used | Output |
|---|---|---|
| Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` | Positive/Negative + confidence % |
| Zero-Shot Classification | `facebook/bart-large-mnli` | Topic category + confidence % |
| Text Summarization | `sshleifer/distilbart-cnn-12-6` | Short abstractive summary |

---

## 🧠 Why These Tasks?

PGAGI works on real AI products. These three tasks represent the core of most production NLP systems:
- **Sentiment** → customer feedback, reviews, social media monitoring
- **Classification** → content routing, auto-tagging, search
- **Summarization** → document processing, news aggregation, reports

---

## 🗂️ Project Structure

```
pgagi-assignment/
├── app.py               # Main pipeline — all 3 NLP tasks
├── requirements.txt     # Dependencies
├── README.md            # This file
└── .gitignore
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/priyanaka-29/pgagi-assignment.git
cd pgagi-assignment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the analyzer
```bash
python app.py
```

### Expected Output
```
🤖  PGAGI AI Text Analyzer — Podugu Priyanka
    Tools: Python | HuggingFace Transformers | NLP

============================================================
  INPUT   : Artificial intelligence is rapidly transforming...
  SENTIMENT : POSITIVE  (99.84% confidence)
  TOPIC     : technology  (94.2% confidence)
  SUMMARY   : Machine learning models are being used in medical
               diagnosis and natural language processing systems.
============================================================
✅  Analysis complete.
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **HuggingFace Transformers** — pre-trained NLP models
- **PyTorch** — deep learning backend
- **GitHub** — version control

---

## 🔗 About Me

- **GitHub:** [github.com/priyanaka-29](https://github.com/priyanaka-29)
- **LinkedIn:** [linkedin.com/in/podugu-priyanka-819a7b382](https://linkedin.com/in/podugu-priyanka-819a7b382)
- **Email:** priyankapodugu21@gmail.com
- **MCA — AI & ML | KL University | CGPA: 9.21**
