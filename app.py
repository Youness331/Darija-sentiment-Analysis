from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib
matplotlib.use('Agg')
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from collections import Counter
import re
import io
import base64
import arabic_reshaper
from bidi.algorithm import get_display
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import csv
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import google.generativeai as genai
import os
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file

from package import dynamic_scraper
from package import scrape_comments

app = Flask(__name__)
import requests



# Load stop words from the CSV file
stop_words_df = pd.read_csv('Stop_words.csv', header=None, encoding='utf-16')
stop_words = set(stop_words_df[0].tolist())  # Convert to a set for faster lookups


# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")

# Load the model structure from the same pre-trained model
model = BertForSequenceClassification.from_pretrained("SI2M-Lab/DarijaBERT", num_labels=2)

# Load the weights from the .safetensors file
safetensors_path = r"C:\Users\dell\Desktop\pfa\test\models\model\model.safetensors"
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

def analyze_sentiment(comments):
    results = []
    all_words = []

    for comment_data in comments:
        comment = comment_data['comment']
        like_count = comment_data['likes']

        # Tokenize the comment with explicit max_length
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=128, padding=True)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).item()

            # Determine sentiment
            sentiment = 'positive' if predictions ==1 else 'negative'
            #print(f"Predicted Sentiment: {sentiment}")
            #print(f"prediction_value: {predictions}")
        # Append result
        results.append({
            'comment': comment,
            'likes': like_count,
            'sentiment': sentiment
        })

        # Tokenize and filter words (remove stopwords)
        words = re.findall(r'\b\w+\b', comment.lower())
        filtered_words = [word for word in words if word not in stop_words]
        all_words.extend(filtered_words)

    # Top 5 common words
    word_counts = Counter(all_words)
    top_5_words = word_counts.most_common(5)

    return results, top_5_words

# Function to load stopwords from stopwords.csv
def load_stopwords(file_path):
    all_stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            all_stopwords.add(row[0].strip())
    arabic_stopwords=stop_words.words('arabic')
    #union arabic with darija stopwords
    all_stopwords= all_stopwords.union(arabic_stopwords)
    return all_stopwords




# Configure the Gemini API
genai.configure(api_key="AIzaSyAMyc31hIwWCwN34IY8ahdhZgYCd50XJ14")

# Function to generate a sentiment report via Gemini-like API
def generate_report(positive_percentage, negative_percentage):
    # Create the prompt for the API using the positive and negative percentages
    prompt = f"The article has {positive_percentage}% positive comments and {negative_percentage}% negative comments. Based on this, generate a sentiment report in arabic summarizing the sentiments presenting  with the articles and by negative comment it means that reader is sad about the article."

    # Use the generative AI model to create a response
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    try:
        response = model.generate_content(prompt)
        # Extract and return the generated report
        return response.text
    
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Could not generate the report due to an API error."

# Function to load stopwords from Stop_words.csv
def load_stopwords(file_path):
    #empty set
    stopwords = set()
    with open(file_path, 'r', encoding='utf-16') as file:
        reader = csv.reader(file)
        for row in reader:
            stopwords.add(row[0].strip())
    return stopwords

# Function to clean and tokenize the text, removing stopwords
def clean_and_tokenize_comments(comments, stop_words):
    all_words = []
    for comment_data in comments:
        comment = comment_data['comment']
        # Tokenize and filter words (remove stopwords)
        words = re.findall(r'\b\w+\b', comment.lower())
        filtered_words = [word for word in words if word not in stop_words]
        all_words.extend(filtered_words)
    return all_words

# Function to generate and display the word cloud
def generate_wordcloud(filtered_words):
    # Create a word cloud
    reshaped_text = arabic_reshaper.reshape(' '.join(filtered_words))
    bidi_text =get_display(reshaped_text)
    wordcloud = WordCloud(font_path='arial', 
                          background_color='white',
                          width=300,
                          height=350).generate(bidi_text)
    # Save wordcloud image to a BytesIO object
    img = io.BytesIO()
    #plot the wordcould image
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Convert image to base64
    wordcloud_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return wordcloud_url


# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to display results after scraping
@app.route('/results', methods=['POST'])
def results():
    key_name = request.form['keyword']
    interval_days = int(request.form['days'])
    url = 'https://www.hespress.com/all?most_viewed'  # URL to scrape from
    
    df = dynamic_scraper(interval_days, key_name, url)
    
    if df is not None:
        # Show the scraped data on the results page
        return render_template('results.html', data=df.to_dict(orient='records'))
    else:
        return render_template('results.html', data=[])


# Route to display all comments and generate the pie chart
@app.route('/all_comments', methods=['POST'])
def all_comments():
    article_links = request.form.getlist('article_links')  # Get all article links from the form

    all_comments = []
    positive_count = 0
    negative_count = 0
    stop_words2 = load_stopwords('Stop_words.csv')
    # Scrape and analyze comments from each article
    for link in article_links:
        comments = scrape_comments(link)
        analyzed_comments, _ = analyze_sentiment(comments)
        all_comments.extend(analyzed_comments)

        # Count positive and negative comments
        for comment in analyzed_comments:
            if comment['sentiment'] == 'positive':
                positive_count += 1
            elif comment['sentiment'] == 'negative':
                negative_count += 1

    # Total number of comments
    total_comments = positive_count + negative_count

    # Calculate percentages (ensure no division by zero)
    if total_comments > 0:
        positive_percentage = (positive_count / total_comments) * 100
        negative_percentage = (negative_count / total_comments) * 100
    else:
        positive_percentage = 0
        negative_percentage = 0

    # Debugging: Print to check calculations
    print(f"Positive Count: {positive_count}, Negative Count: {negative_count}")
    print(f"Positive Percentage: {positive_percentage}%, Negative Percentage: {negative_percentage}%")

    # Generate the sentiment distribution pie chart (ensure total_comments > 0 to avoid NaN)
    if total_comments > 0:
        labels = ['Positive', 'Negative']
        sizes = [positive_percentage, negative_percentage]
        colors = ['#66b3ff', '#ff9999']
        
        # Create pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object and convert it to a base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
    else:
        chart_url = None  # If no comments, no chart is generated
    sentiment_report = generate_report(positive_percentage, negative_percentage)
    # Generate the word cloud from all comments
    filtered_words = clean_and_tokenize_comments(all_comments, stop_words2)
    wordcloud_url = generate_wordcloud(filtered_words)

    # Pass data to the template
    return render_template(
        'all_comments.html',
        sentiment_results=all_comments,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_percentage=positive_percentage,
        negative_percentage=negative_percentage,
        chart_url=chart_url,
        sentiment_report=sentiment_report,
        wordcloud_url=wordcloud_url
    )



@app.route('/analyze_comments/<path:article_link>', methods=['GET'])
def analyze_comments(article_link):
    comments = scrape_comments(article_link)  # Scrape comments
    
    # Debugging: Check if comments are scraped
    print("Scraped Comments:", comments)
    
    if not comments:
        print("No comments found")
    
    sentiment_results, top_5_words = analyze_sentiment(comments)  # Analyze sentiment and get top 5 redundant words
    
    # Print the top 5 redundant words for debugging
    print("Top 5 Redundant Words:", top_5_words)
    
    return render_template('comments.html', sentiment_results=sentiment_results, top_5_words=top_5_words)



if __name__ == '__main__':
    app.run(debug=True)