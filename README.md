# 📊 YouTube Comment Sentiment Analysis Pipeline

This project performs sentiment analysis and feature engineering on YouTube comments, merges them with video engagement metrics, and trains various classification models to predict like/dislike ratio.

## 🔧 Features

- Cleans and tokenizes YouTube comments  
- Embeds text using GloVe word vectors  
- Scores sentiment using VADER  
- Merges in dislike and view data for contextual modeling  
- Supports binary and multiclass classification  
- Trains models including Decision Tree, Random Forest, Gradient Boosting, and XGBoost  
- Generates word clouds for sentiment visualization  
- Configurable pipeline stages with caching and logging  

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/yt-comments-pipeline.git
cd yt-comments-pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

```bash
python main.py --config config.yaml
```

To only retrain models on existing features:

```bash
python main.py --retrain-only
```

### 4. Get the Data
This project relies on YouTube comment and engagement datasets. 
You can reach out directly if you'd like a copy of the data files.
Alternatively, continue reading to learn how to run the project on docker with the given data.
---

## ⚙️ Configuration

Edit `config.yaml` to toggle pipeline steps and specify behavior:

```yaml
data_dir: data
classification_type: binary  # or "multiclass"
use_smote: true
models:
  - random_forest
  - xgboost
recompute:
  load_clean_tokenize: false
  embed: false
  merge_dislikes: false
  sentiment: false
  features: false
  train: true
```

---

## 🐳 Running with Docker

Build the container:

```bash
docker build -t yt-comments-pipeline .
```

Run it:

```bash
docker run --rm -v $(pwd)/data:/app/data yt-comments-pipeline
```

Running it from the Docker Registry:

```bash
docker run --rm schwandersc/yt-comments-pipeline:latest
```

And you can mount your own config file/data directory

```bash
docker run --rm \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/data:/app/data \
  your_dockerhub_username/yt-comments-pipeline:latest \
  --config config.yaml
```


---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📚 Generating Documentation

```bash
pdoc -d google -o docs src
```

View locally:

```bash
python3 -m http.server 8080 --directory docs
```

---

## 🗂 Project Structure

```
.
├── src/
│   ├── clean.py
│   ├── embed.py
│   ├── feature_engineering.py
│   ├── load.py
│   ├── sentiment.py
│   ├── train.py
│   └── visualize.py
├── data/
├── logs/
├── tests/
├── main.py
├── config.yaml
├── requirements.txt
└── Dockerfile
```

---

## 📌 License

This project is MIT licensed. Feel free to use, fork, and adapt it.
