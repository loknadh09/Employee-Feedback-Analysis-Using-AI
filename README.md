🧠 Employee Feedback Analysis Using AI

This project is an AI-powered Employee Feedback Analysis System built using Flask, NLP, and a machine-learning–backed sentiment classifier.
Users can paste multiple feedback comments, and the system generates:

Individual sentiment labels (Positive / Negative / Neutral)

Polarity scores

Keyword extraction

Overall sentiment statistics

Interactive dashboard visualizations

Word cloud representation

The app uses a hybrid ML + rule-based sentiment engine, ensuring it works even if the trained model is unavailable.
Backend powered by Flask; NLP powered by scikit-learn + custom preprocessing.

🚀 Features
🔹 1. Web Dashboard for Feedback Analysis

The main app (app.py) provides a browser interface and performs:

Sentiment analysis for each comment 

app

Keyword extraction (via CountVectorizer)

Aggregated sentiment stats (positive / negative / neutral %)

Rendering of charts + word cloud UI

🔹 2. NLP Engine

The NLP logic (nlp_utils.py) implements:

Text cleaning & normalization

Model loading with auto-fallback to rule-based mode

Rule-based sentiment using positive/negative lexicons

ML classifier prediction if sentiment_model.pkl is available 

nlp_utils

Keyword extraction using TF-based word counts

🔹 3. Trainable Machine Learning Model

The model training script (train_model.py) provides:

Automatic detection of text and sentiment/rating columns

Converts rating → sentiment if needed

Trains TF-IDF + Logistic Regression model

Saves sentiment_model.pkl for the web app 

smoke_test_client

Avoids crashes with small datasets

Generates classification metrics

🔹 4. Smoke Test Script

smoke_test_client.py runs automated tests against the Flask app.
It posts sample feedback and saves the rendered HTML response locally.
Used to verify UI rendering and backend functionality. 

smoke_test_response

📁 Project Structure
Employee-Feedback-Analysis-Using-AI/
│
├── app.py                     # Flask web app backend
├── nlp_utils.py               # NLP + sentiment engine
├── train_model.py             # Model training pipeline
├── sentiment_model.pkl        # Saved ML model (optional)
├── smoke_test_client.py       # Automated backend tester
├── smoke_test_response.html   # Test output snapshot
├── templates/
│   └── index.html             # Dashboard UI & charts
└── README.md


🛠️ Technologies Used

Python 3

Flask

scikit-learn

Pandas / NumPy

TfidfVectorizer & LogisticRegression

CountVectorizer (keyword extraction)

HTML / CSS dashboard interface

📊 How It Works

User submits multiline feedback

Feedback is split into individual comments

Each comment is cleaned and analyzed

Model predicts sentiment

If model missing or errors → rule-based fallback gracefully

Keywords extracted using CountVectorizer

Dashboard displays:

Sentiment percentages

Bar chart

Satisfaction chart

Word cloud

Detailed table of results

Backend logic processed in app.py, sentiment in nlp_utils.py.
Model training uses train_model.py.

▶️ Running the Application
1. Install dependencies
pip install -r requirements.txt


(If you don’t have one, I can generate it.)

2. Run the Flask app
python app.py

3. Open in browser
http://127.0.0.1:5000

🧪 Training Your Own Sentiment Model

If you want stronger predictions, place a dataset in CSV format and run:

python train_model.py


The script will:

Detect text column automatically

Convert rating → sentiment if needed

Train TF-IDF + Logistic Regression

Save sentiment_model.pkl automatically

The web app will use this model on the next run.

🔮 Future Enhancements

Deploy to Cloud (Render / AWS / Azure / Railway)

Add database storage for feedback logs

Improve word cloud using actual frequency visualization

Add interactive charts (Plotly)

Add topic modeling (LDA / BERTopic)

👨‍💻 Author

Loknadh
GitHub: https://github.com/loknadh09
