# 📄 Task 1: Customer Support Ticket Classification & Entity Extraction

## 🎯 Objective

Develop a machine learning pipeline that:
- Classifies customer support tickets by:
  - **Issue Type**
  - **Urgency Level**
- Extracts entities from the ticket:
  - **Product Name**
  - **Dates**
  - **Complaint Keywords**

---

## 📦 Dataset: ai_dev_assignment_tickets_complex_1000.xlsx

| Column           | Description                                 |
|------------------|---------------------------------------------|
| `ticket_id`      | Unique ID for the ticket                    |
| `ticket_text`    | Text of the customer support request        |
| `issue_type`     | Label for multi-class classification        |
| `urgency_level`  | Label for urgency (Low, Medium, High)       |
| `product`        | Ground truth for entity extraction          |

---

## 🛠️ Pipeline Components

### 🔹 1. Data Preprocessing
- Lowercasing, punctuation and digit removal
- Stopword removal using NLTK
- Lemmatization
- Dropped rows with missing values

### 🔹 2. Feature Engineering
- TF-IDF (top 300 features)
- Ticket length (word count)
- Sentiment score using TextBlob

### 🔹 3. Multi-Task Classification
- Model 1: Issue Type (Logistic Regression) → ✅ ~100% accuracy
- Model 2: Urgency Level (Logistic Regression) → ⚠️ ~33% accuracy
- Evaluation: Classification Report + Confusion Matrix

### 🔹 4. Entity Extraction (Rule-Based)
- Extract product from ticket text using known list
- Extract dates using regex
- Extract complaint keywords from predefined list

### 🔹 5. Integration Function
`analyze_ticket(text)`:
- Predicts issue type & urgency level
- Returns extracted entities (product, dates, complaints)

### 🔹 6. Gradio Interface
- Takes raw ticket text as input
- Displays:
  - Predicted issue type
  - Predicted urgency level
  - Extracted product name
  - Full extracted entities (in JSON format)

---

## 📊 Evaluation Summary

| Task              | Accuracy / Performance |
|------------------|-------------------------|
| Issue Type        | ✅ 100% (separable labels) |
| Urgency Level     | ⚠️ ~33% (requires tuning) |
| Product Extraction | ✅ Approx. `XX%` match with ground truth |

---

## ▶️ How to Run

### 1. Install requirements:
```bash
pip install pandas scikit-learn nltk textblob gradio
