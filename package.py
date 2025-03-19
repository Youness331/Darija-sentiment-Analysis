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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import csv
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import emoji
# Ensure NLTK stopwords are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

arabic_to_english_day = {
        'السبت': 'Saturday', 'الأحد': 'Sunday', 'الإثنين': 'Monday',
        'الثلاثاء': 'Tuesday', 'الأربعاء': 'Wednesday', 'الخميس': 'Thursday', 'الجمعة': 'Friday'
    }
arabic_to_english_month = {
        'يناير': 'January', 'فبراير': 'February', 'مارس': 'March', 'أبريل': 'April',
        'مايو': 'May', 'يونيو': 'June', 'يوليوز': 'July', 'غشت': 'August', 'شتنبر': 'September',
        'أكتوبر': 'October', 'نونبر': 'November', 'دجنبر': 'December'
    }

def dynamic_scraper(interval_days, key_name, url):
    # Calculate the date interval
    end_date = datetime.now()
    start_date = end_date - timedelta(days=interval_days)    
    # Initialize WebDriver
    chrome_service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=chrome_service)
    driver.get(url)
    time.sleep(5)  # Allow page to load
    
    data = []
    seen_data = set()
    last_height = driver.execute_script("return document.body.scrollHeight")
    finished_scraping = False  
    
    while not finished_scraping:
        titles = driver.find_elements(By.CSS_SELECTOR, 'h3.card-title')
        times = driver.find_elements(By.CSS_SELECTOR, 'small.text-muted.time')
        links = driver.find_elements(By.CSS_SELECTOR, 'a.stretched-link')
        
        if len(titles) == len(times) == len(links):
            for i in range(len(titles)):
                title = titles[i].text.strip()
                time_str = times[i].text.strip()
                link = links[i].get_attribute('href')
                
                for arabic_day, english_day in arabic_to_english_day.items():
                    time_str = time_str.replace(arabic_day, english_day)
                for arabic_month, english_month in arabic_to_english_month.items():
                    time_str = time_str.replace(arabic_month, english_month)
                
                try:
                    date_obj = datetime.strptime(time_str, '%A %d %B %Y - %H:%M')
                except ValueError:
                    continue
                
                unique_key = (date_obj, title)
                if start_date <= date_obj <= end_date and key_name in title and unique_key not in seen_data:
                    data.append({'date': time_str, 'title': title, 'link': link})
                    seen_data.add(unique_key)
        
        if times:
            last_time_str = times[-1].text.strip()
            for arabic_day, english_day in arabic_to_english_day.items():
                last_time_str = last_time_str.replace(arabic_day, english_day)
            for arabic_month, english_month in arabic_to_english_month.items():
                last_time_str = last_time_str.replace(arabic_month, english_month)
            
            last_date_obj = datetime.strptime(last_time_str, '%A %d %B %Y - %H:%M')
            if last_date_obj <= start_date:
                finished_scraping = True
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1000);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            finished_scraping = True
        last_height = new_height
        time.sleep(5)
    
    driver.quit()
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(f'{key_name}_scraped_data.csv', index=False)
        return df
    else:
        print("No data found within the specified date range")
        return None

#fonction pour le scraping des commentaires
def scrape_comments(article_link):
    try:
        response = requests.get(article_link)
        
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # Print the HTTP error
        return []  # Return an empty list or handle the error as needed
    except Exception as err:
        print(f"Other error occurred: {err}")  # Print any other error
        return []  # Return an empty list or handle the error as needed
    #wait 3 secondes to request
    time.sleep(3)
    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.find_all('div', class_='comment-text')
    likes = soup.find_all('span', class_='comment-recat-number')

    comments_list = []
    for i in range(len(comments)):
        comment_text = comments[i].get_text(strip=True)
        try:
            like_count = likes[i].get_text(strip=True)
        except IndexError:
            like_count = '0'
        comments_list.append({'comment': comment_text, 'likes': like_count})
    return comments_list

#fonction pour scraper les infos des articles
def scrape_title_categorie_date_author_summary(link):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(link, headers=headers)
        response.raise_for_status()  

        soup = BeautifulSoup(response.content, 'html.parser')

        # Title
        title_element = soup.find('h1', class_='post-title')
        article_title = title_element.text.strip() if title_element else ''

        # Author
        author_span = soup.find('span', class_='author')
        author = author_span.text.strip() if author_span else ''

        # Date
        date_span = soup.find('span', class_='date-post')
        date = date_span.text.strip() if date_span else ''

        # Category 
        breadcrumb_elements = soup.find_all('li', class_='breadcrumb-item')
        category_text = ''
        if len(breadcrumb_elements) > 1:
            category_text = breadcrumb_elements[1].text.strip()

        # Summary 
        summary_element = soup.find('div', class_='article-content')
        if summary_element:
            if summary_element.find('iframe'):  # If summary contains an iframe 
                summary = summary_element.find('iframe')['src']
            else:
                summary = summary_element.text.strip()
        else:
            summary = ''

        return article_title, category_text, date, author, summary
    
    except Exception as e:
        print(f"Error scraping data from {link}")
        return None, None, None, None, None

#fonction pour cleaning du comments
def clean_text(text):
    stopwords_df = pd.read_csv('Stop_words .csv', header=None, names=['stopword'],encoding='utf-16')
    darija_stopwords = set(stopwords_df['stopword'])
    arabic_stopwords = set(stopwords.words('arabic'))
     #union darija with arabic stop words
    stopwords = darija_stopwords.union(arabic_stopwords)
    emoji_pattern = re.compile(
    "[\U0001F600-\U0001F64F]"  # emoticons
    "|[\U0001F300-\U0001F5FF]"  # symbols & pictographs
    "|[\U0001F680-\U0001F6FF]"  # transport & map symbols
    "|[\U0001F1E0-\U0001F1FF]"  # flags (iOS)
    "|[\U00002702-\U000027B0]"  # other symbols
    "|[\U000024C2-\U0001F251]"  # enclosed characters
    "+", flags=re.UNICODE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    #Normalize Arabic text
    # Remove Harakat (diacritics)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Normalize different forms of Arabic letters
    text = re.sub(r'[أإآ]', 'ا', text)  # Normalize 'أ', 'إ', 'آ' to 'ا'
    text = re.sub(r'ُ|ُ|ٰ|ۥ', 'و', text)  
     # Remove numbers
    text = re.sub(r'\d+', '', text)
     
    
    # Remove repetitive letters 
    text = re.sub(r'(.)\1+', r'\1\1', text)
    #remove repetitive emojies to only one emoji
    text = emoji_pattern.sub(lambda x: x.group(0)[0], text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords]

    return ' '.join(filtered_tokens)
