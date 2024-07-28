from flask import Flask, request, render_template
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import io
import base64
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk

app = Flask(__name__)

# Stopword Packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

def read_list_from_txt(file):
    return [line.strip() for line in file.read().decode('utf-8').splitlines() if line.strip()]

def remove_stop_words_and_punctuation(text):
    tokens = word_tokenize(text)
    filtered_tokens = [
        word.lower() for word in tokens
        if word.lower() not in stop_words and word.isalpha()
    ]
    return ' '.join(filtered_tokens)

def extract_keywords_from_reviews(reviews):
    # Remove stop words and punctuation from reviews
    processed_reviews = [remove_stop_words_and_punctuation(review.lower()) for review in reviews]
    
    # Tokenize the processed reviews
    texts = [review.split() for review in processed_reviews]
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train LDA model
    num_topics = min(5, len(dictionary))  # Increase the number of topics
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.show_topics(num_words=30, formatted=False)
    
    # Extract keywords
    keywords = set()
    for topic in topics:
        for word, _ in topic[1]:
            keywords.add(word)
    
    # Remove common words using frequency analysis
    all_words = [word for text in texts for word in text]
    fdist = FreqDist(all_words)
    
    # Minimum Frequency
    min_freq = 50
    keywords = {word for word in keywords if fdist[word] >= min_freq}
    
    tagged_keywords = pos_tag(keywords)
    keywords = {word for word, pos in tagged_keywords if pos in ('NN', 'NNS', 'JJ', 'VB')}
    
    return list(keywords)

# Count frequency of words
def count_mentions(reviews, items):
    item_counts = {item: 0 for item in items}
    for review in reviews:
        text = review.lower()
        for item in items:
            if item.lower() in text:
                item_counts[item] += 1
    return item_counts

# Perform sentiment analysis
def analyze_sentiment(text, model=None):
    if model:
        return model.predict([text])[0]
    else:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

# If labels are provided
def train_sentiment_model(labeled_data):
    X = labeled_data['text']
    y = labeled_data['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create a pipeline with a vectorizer and a classifier
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Train the model
    pipeline.fit(X, y_encoded)
    
    return pipeline, label_encoder

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    items_file = request.files.get('items_file')
    reviews_file = request.files['reviews_file']
    labels_file = request.files.get('labels_file')

    # Load reviews
    reviews_data = read_list_from_txt(reviews_file)
    reviews = [review.lower() for review in reviews_data]

    # Load item list if provided
    if items_file:
        items = read_list_from_txt(items_file)
    else:
        # If the user doesn't have a topic list, extract topics
        items = extract_keywords_from_reviews(reviews)

    # Initialize model and label encoder
    custom_model = None
    label_encoder = None

    if labels_file:
        # Load labeled data and train a custom sentiment model
        labeled_data = pd.read_csv(labels_file, delimiter=',')
        custom_model, label_encoder = train_sentiment_model(labeled_data)

    # Analyze sentiment
    item_counts = count_mentions(reviews, items)
    item_sentiments = {item: [] for item in items}

    for review in reviews:
        for item in items:
            if item.lower() in review:
                sentiment_score = analyze_sentiment(review, custom_model)
                item_sentiments[item].append(sentiment_score)

    # Rank items based on frequency and sentiment
    item_ranking = {}
    for item in items:
        mentions = item_counts[item]
        if item_sentiments[item]:
            avg_sentiment = sum(item_sentiments[item]) / len(item_sentiments[item])
        else:
            avg_sentiment = 0
        rank_score = mentions * avg_sentiment
        item_ranking[item] = rank_score

    sorted_items = sorted(item_ranking.items(), key=lambda x: x[1], reverse=True)
    sorted_items_names = [item[0] for item in sorted_items[:29]]
    sorted_items_scores = [item[1] for item in sorted_items[:29]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_items_names, sorted_items_scores, color='skyblue')
    ax.set_xlabel('Overall Sentiment')
    ax.set_title('Ranking of Items based on Frequency and Sentiment')
    ax.invert_yaxis()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return render_template('index.html', plot_url='data:image/png;base64,' + plot_url)

if __name__ == '__main__':
    app.run(debug=True)
