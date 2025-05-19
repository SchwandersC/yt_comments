# 📊 YouTube Comment Sentiment Analysis Pipeline

## 🧠 Motivation
In 2019, YouTube made the controversial decision to remove public dislike counts from its videos. While the goal was to reduce hate-driven behavior and spam, the change also removed an important mechanism for community feedback and content accountability.

Today, if someone wants to gauge how a video has been received, they must dig through the comment section—a time-consuming and often biased process.

With the rise of advanced natural language processing and machine learning techniques, this process can now be automated. By analyzing the sentiment and patterns in viewer comments, it's possible to quickly estimate a video's trustworthiness and reception—saving users time and potentially protecting them from misinformation or harmful content.

## 🧪 Project Overview
This project applies machine learning to predict the like-to-dislike ratio of YouTube videos based on user comments and engagement metrics. It has two main components:

### Model Training & Analysis
A training pipeline that takes in a curated dataset of YouTube videos with known dislike counts, performs sentiment analysis, and trains classification models to learn engagement patterns.

### Interactive Web App
A Flask-powered web app where users can input any YouTube video ID and receive a predicted like-to-dislike ratio, powered by the trained models. The app is currently hosted here https://ytcomments-api-278551739864.us-central1.run.app 
(Caution: App may take a minute or so to load due to cloud run's cold start container app solutions)

## 🔧 Features

- Cleans and tokenizes YouTube comments  
- Embeds text using GloVe word vectors  
- Scores sentiment using VADER  
- Merges in dislike and view data for contextual modeling  
- Supports binary and multiclass classification  
- Trains models including Decision Tree, Random Forest, Gradient Boosting, and XGBoost  
- Generates word clouds for sentiment visualization  
- Configurable pipeline stages with caching and logging  
- **Lightweight web interface with `app.py` for on-demand inference**  
- **Docker-first architecture for reproducible builds and easier deployment**

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

---

## 🧠 Web Inference App

The project includes a minimal `app.py` interface, which allows users to input comments and receive predicted sentiment or popularity class directly in a browser or via REST API.

### Run the app locally:

```bash
python app.py
```

The server will start at `http://localhost:5000`.

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

You can now run the full pipeline or the web app using Docker for consistency and portability.

### Build the container:

```bash
docker build -t yt-comments-pipeline .
```

### Run pipeline:

```bash
docker run --rm -v $(pwd)/data:/app/data yt-comments-pipeline
```

### Run the web app:

```bash
docker run -p 5000:5000 yt-comments-pipeline python app.py
```

This will expose the app at `http://localhost:5000`.

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/wyLxvtEMfV8/0.jpg)](https://www.youtube.com/watch?v=wyLxvtEMfV8)

Click the image above to watch a walkthrough of the pipeline and web app.


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
├── app.py
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
