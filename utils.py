"""
Utility functions for news extraction, sentiment analysis, and text-to-speech conversion.
"""

import re
import json
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple, Optional
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from gtts import gTTS
import nltk
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

# News Extraction Functions
def search_news_articles(company_name: str, num_articles: int = 15) -> List[Dict[str, Any]]:
    """
    Search for news articles related to a company using Google News.
    
    Args:
        company_name: Name of the company to search for
        num_articles: Number of articles to retrieve (default: 15)
        
    Returns:
        List of dictionaries containing article URLs
    """
    logger.info(f"Searching for news articles about {company_name}")
    
    articles = []
    
    try:
        # Format the search query for Google News
        query = f"{company_name} company news"
        query = query.replace(' ', '+')
        
        # Google News URL
        url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Send request to Google News
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article elements
            article_elements = soup.select('article')
            
            for article in article_elements:
                if len(articles) >= num_articles:
                    break
                
                # Extract article link
                link_element = article.select_one('a[href^="./article"]')
                if not link_element:
                    continue
                
                # Get the relative URL and convert to absolute URL
                relative_url = link_element['href']
                if relative_url.startswith('./'):
                    relative_url = relative_url[2:]  # Remove './' prefix
                
                article_url = f"https://news.google.com/{relative_url}"
                
                # Extract source
                source_element = article.select_one('div[data-n-tid="9"]')
                source = source_element.text if source_element else "Unknown Source"
                
                articles.append({
                    "url": article_url,
                    "source": source
                })
            
            logger.info(f"Found {len(articles)} articles from Google News")
        else:
            logger.warning(f"Failed to fetch news from Google News: Status code {response.status_code}")
    
    except Exception as e:
        logger.error(f"Error searching for news articles: {e}")
    
    # If we couldn't get enough articles, try alternative sources
    if len(articles) < num_articles:
        try:
            # Try another source like Bing News
            query = f"{company_name} company news"
            query = query.replace(' ', '+')
            
            url = f"https://www.bing.com/news/search?q={query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find news cards
                news_cards = soup.select('.news-card')
                
                for card in news_cards:
                    if len(articles) >= num_articles:
                        break
                    
                    link_element = card.select_one('a.title')
                    if not link_element or 'href' not in link_element.attrs:
                        continue
                    
                    article_url = link_element['href']
                    
                    # Extract source
                    source_element = card.select_one('.source')
                    source = source_element.text if source_element else "Unknown Source"
                    
                    # Check if this URL is already in our list
                    if not any(article['url'] == article_url for article in articles):
                        articles.append({
                            "url": article_url,
                            "source": source
                        })
                
                logger.info(f"Found additional {len(articles)} articles from Bing News")
            else:
                logger.warning(f"Failed to fetch news from Bing News: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error searching for additional news articles: {e}")
    
    # Try a third source if we still don't have enough articles
    if len(articles) < num_articles:
        try:
            # Try Yahoo Finance
            query = f"{company_name}"
            query = query.replace(' ', '%20')
            
            url = f"https://finance.yahoo.com/quote/{query}/news"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find news links
                news_links = soup.select('a[href^="/news/"]')
                
                for link in news_links:
                    if len(articles) >= num_articles:
                        break
                    
                    if 'href' not in link.attrs:
                        continue
                    
                    relative_url = link['href']
                    article_url = f"https://finance.yahoo.com{relative_url}"
                    
                    # Check if this URL is already in our list
                    if not any(article['url'] == article_url for article in articles):
                        articles.append({
                            "url": article_url,
                            "source": "Yahoo Finance"
                        })
                
                logger.info(f"Found additional {len(articles)} articles from Yahoo Finance")
            else:
                logger.warning(f"Failed to fetch news from Yahoo Finance: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error searching for additional news articles from Yahoo Finance: {e}")
    
    # Ensure we don't return more than requested
    articles = articles[:num_articles]
    
    logger.info(f"Returning {len(articles)} articles for analysis")
    return articles

def extract_article_content(url: str, source: str) -> Dict[str, Any]:
    """
    Extract content from a news article URL using BeautifulSoup.
    
    Args:
        url: URL of the article
        source: Source of the article
        
    Returns:
        Dictionary containing article details
    """
    logger.info(f"Extracting content from {url}")
    
    # Initialize default values
    title = ""
    content = ""
    summary = ""
    article_date = ""
    
    try:
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Send request to the article URL
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title - try different common patterns
            if not title:
                title_element = soup.find('h1') or soup.find('title')
                if title_element:
                    title = title_element.text.strip()
            
            # Extract date - try different common patterns
            date_elements = soup.select('time') or soup.select('[datetime]') or soup.select('.date, .published, .publish-date, .timestamp')
            if date_elements:
                article_date = date_elements[0].text.strip()
                # If date is empty but has datetime attribute
                if not article_date and 'datetime' in date_elements[0].attrs:
                    article_date = date_elements[0]['datetime'].split('T')[0]
            
            # If no date found, use current date
            if not article_date:
                import datetime
                article_date = datetime.datetime.now().strftime("%B %d, %Y")
            
            # Extract content - try different common patterns
            content_elements = soup.select('article') or soup.select('.article-body') or soup.select('.story-body')
            
            if content_elements:
                # Get all paragraphs from the content element
                paragraphs = content_elements[0].find_all('p')
                content = ' '.join([p.text.strip() for p in paragraphs])
            else:
                # Fallback: get all paragraphs from the body
                paragraphs = soup.find_all('p')
                # Filter out short paragraphs that are likely not part of the main content
                paragraphs = [p for p in paragraphs if len(p.text.strip()) > 50]
                content = ' '.join([p.text.strip() for p in paragraphs])
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Create a summary (first 2-3 sentences)
            sentences = content.split('.')
            summary = '.'.join(sentences[:min(3, len(sentences))]) + '.'
            
            logger.info(f"Successfully extracted content from {url}")
        else:
            logger.warning(f"Failed to fetch article content: Status code {response.status_code}")
            return {
                "title": "",
                "content": "",
                "summary": "",
                "url": url,
                "source": source,
                "date": "",
                "sentiment": analyze_sentiment(""),
                "topics": []
            }
    
    except Exception as e:
        logger.error(f"Error extracting article content: {e}")
        return {
            "title": "",
            "content": "",
            "summary": "",
            "url": url,
            "source": source,
            "date": "",
            "sentiment": analyze_sentiment(""),
            "topics": []
        }
    
    # If we couldn't extract meaningful content, return empty content
    if not title or not content or len(content) < 100:
        logger.warning(f"Extracted content was insufficient")
        return {
            "title": title or "",
            "content": "",
            "summary": "",
            "url": url,
            "source": source,
            "date": article_date or "",
            "sentiment": analyze_sentiment(""),
            "topics": []
        }
    
    # Analyze sentiment
    sentiment_data = analyze_sentiment(content)
    
    # Extract topics
    topics = extract_topics(content)
    
    return {
        "title": title,
        "content": content,
        "summary": summary,
        "url": url,
        "source": source,
        "date": article_date,
        "sentiment": sentiment_data,
        "topics": topics
    }

# Initialize sentiment analysis model
sentiment_analyzer = None

def get_sentiment_analyzer():
    """
    Get or initialize the sentiment analysis model.
    
    Returns:
        Transformer pipeline for sentiment analysis
    """
    global sentiment_analyzer
    if sentiment_analyzer is None:
        try:
            # Initialize the transformer-based sentiment analysis model
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True
            )
            logger.info("Initialized transformer-based sentiment analysis model")
        except Exception as e:
            logger.error(f"Error initializing transformer model: {e}")
            # Fall back to VADER if transformer model fails
            logger.info("Falling back to VADER for sentiment analysis")
            sentiment_analyzer = "vader"
    
    return sentiment_analyzer

# Sentiment Analysis Functions
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of a text using a transformer-based model.
    Falls back to VADER if the transformer model is unavailable.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary containing sentiment scores and category
    """
    if not text:
        return {
            "compound": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "category": "Neutral"
        }
    
    try:
        analyzer = get_sentiment_analyzer()
        
        if analyzer == "vader":
            # Use VADER as fallback
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = sid.polarity_scores(text)
            
            # Determine sentiment category
            compound = sentiment_scores['compound']
            if compound >= 0.05:
                category = "Positive"
            elif compound <= -0.05:
                category = "Negative"
            else:
                category = "Neutral"
            
            return {
                "compound": sentiment_scores['compound'],
                "positive": sentiment_scores['pos'],
                "negative": sentiment_scores['neg'],
                "neutral": sentiment_scores['neu'],
                "category": category
            }
        else:
            # Use transformer model
            # Limit text length to avoid issues with long articles
            # Most transformer models have a token limit (e.g., 512 tokens)
            # We'll use the first 1000 characters as a representative sample
            sample_text = text[:1000]
            
            # Get sentiment prediction
            result = analyzer(sample_text)[0]
            label = result['label']
            score = result['score']
            
            # Map to our expected format
            if label == "POSITIVE":
                category = "Positive"
                positive_score = score
                negative_score = 1 - score
                neutral_score = 0.0
                compound_score = score * 2 - 1  # Scale from [0,1] to [-1,1]
            elif label == "NEGATIVE":
                category = "Negative"
                positive_score = 1 - score
                negative_score = score
                neutral_score = 0.0
                compound_score = -score * 2 + 1  # Scale from [0,1] to [-1,1]
            else:
                # This shouldn't happen with the default model, but just in case
                category = "Neutral"
                positive_score = 0.5
                negative_score = 0.5
                neutral_score = 0.0
                compound_score = 0.0
            
            return {
                "compound": compound_score,
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score,
                "category": category,
                "model": "transformer"  # Add this to indicate we used the advanced model
            }
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        
        # Try VADER as a fallback if transformer fails
        try:
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = sid.polarity_scores(text)
            
            # Determine sentiment category
            compound = sentiment_scores['compound']
            if compound >= 0.05:
                category = "Positive"
            elif compound <= -0.05:
                category = "Negative"
            else:
                category = "Neutral"
            
            return {
                "compound": sentiment_scores['compound'],
                "positive": sentiment_scores['pos'],
                "negative": sentiment_scores['neg'],
                "neutral": sentiment_scores['neu'],
                "category": category,
                "model": "vader_fallback"  # Indicate we fell back to VADER
            }
        except:
            # If all else fails, return neutral
            return {
                "compound": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 1,
                "category": "Neutral",
                "model": "none"  # Indicate no model was used
            }

def extract_topics(text: str, n_topics: int = 5) -> List[str]:
    """
    Extract main topics from text using TF-IDF.
    
    Args:
        text: Text to analyze
        n_topics: Number of topics to extract
        
    Returns:
        List of topic keywords
    """
    if not text or len(text.split()) < 5:
        return []
    
    try:
        # Tokenize and clean text
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 3]
        
        # If we have very few words, return them directly
        if len(words) <= n_topics:
            return words
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=n_topics)
        tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
        
        # Get feature names and their scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Sort by score and return top n_topics
        topics = [feature_names[i] for i in scores.argsort()[-n_topics:][::-1]]
        return topics
    
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return []

# Comparative Analysis Functions
def perform_comparative_analysis(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform comparative analysis across multiple articles.
    
    Args:
        articles: List of article dictionaries with sentiment and topics
        
    Returns:
        Dictionary containing comparative analysis results
    """
    if not articles:
        return {
            "sentiment_distribution": {},
            "coverage_differences": [],
            "topic_overlap": {}
        }
    
    try:
        # Sentiment distribution
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for article in articles:
            if "sentiment" in article and "category" in article["sentiment"]:
                sentiment_counts[article["sentiment"]["category"]] += 1
        
        # Topic analysis
        all_topics = []
        for article in articles:
            if "topics" in article:
                all_topics.extend(article["topics"])
        
        # Count topic frequency
        topic_counts = {}
        for topic in all_topics:
            if topic in topic_counts:
                topic_counts[topic] += 1
            else:
                topic_counts[topic] = 1
        
        # Find common and unique topics
        common_topics = [topic for topic, count in topic_counts.items() if count > 1]
        unique_topics = [topic for topic, count in topic_counts.items() if count == 1]
        
        # Generate coverage differences
        coverage_differences = []
        
        # Compare articles with different sentiments
        positive_articles = [a for a in articles if a.get("sentiment", {}).get("category") == "Positive"]
        negative_articles = [a for a in articles if a.get("sentiment", {}).get("category") == "Negative"]
        
        if positive_articles and negative_articles:
            coverage_differences.append({
                "comparison": f"Articles with positive sentiment focus on different aspects than negative ones.",
                "impact": "This contrast highlights the polarized coverage of the company."
            })
        
        # Compare articles from different sources
        sources = set(article["source"] for article in articles)
        if len(sources) > 1:
            coverage_differences.append({
                "comparison": f"Coverage varies across {', '.join(sources)}.",
                "impact": "Different news sources emphasize different aspects of the company."
            })
        
        # Final sentiment analysis
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        final_sentiment = f"The company's news coverage is predominantly {dominant_sentiment.lower()}."
        
        return {
            "sentiment_distribution": sentiment_counts,
            "coverage_differences": coverage_differences,
            "topic_overlap": {
                "common_topics": common_topics,
                "unique_topics": unique_topics
            },
            "final_sentiment_analysis": final_sentiment
        }
    
    except Exception as e:
        logger.error(f"Error performing comparative analysis: {e}")
        return {
            "sentiment_distribution": {},
            "coverage_differences": [],
            "topic_overlap": {}
        }

# Text-to-Speech Functions
def generate_hindi_tts(text: str, output_file: str = "output.mp3") -> str:
    """
    Generate Hindi text-to-speech audio.
    
    Args:
        text: Text to convert to speech (in Hindi)
        output_file: Path to save the audio file
        
    Returns:
        Path to the generated audio file
    """
    try:
        # Ensure the text is in Hindi or translate it
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(output_file)
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating Hindi TTS: {e}")
        return ""

def translate_to_hindi(text: str) -> str:
    """
    Translate text to Hindi using a simple approach.
    Note: For production, consider using a proper translation API.
    
    Args:
        text: English text to translate
        
    Returns:
        Hindi translated text
    """
    # This is a placeholder. In a real implementation, you would use a translation API
    # For now, we'll use a few hardcoded translations for demonstration
    translations = {
        "positive": "सकारात्मक",
        "negative": "नकारात्मक",
        "neutral": "तटस्थ",
        "sentiment": "भावना",
        "analysis": "विश्लेषण",
        "news": "समाचार",
        "article": "लेख",
        "company": "कंपनी"
    }
    
    # Very basic word replacement (not a real translation)
    translated = text
    for eng, hindi in translations.items():
        translated = translated.replace(eng, hindi)
    
    return translated

# Process a complete company analysis
def process_company_news(company_name: str, num_articles: int = 10) -> Dict[str, Any]:
    """
    Process news articles for a company and generate a complete analysis.
    
    Args:
        company_name: Name of the company
        num_articles: Number of articles to analyze
        
    Returns:
        Dictionary containing complete analysis results
    """
    logger.info(f"Processing news for {company_name}")
    
    try:
        # Search for news articles
        article_urls = search_news_articles(company_name, num_articles)
        
        # Extract content from each article
        articles = []
        for article_info in article_urls:
            article_data = extract_article_content(article_info["url"], article_info["source"])
            
            # Skip articles with no content
            if not article_data["content"]:
                continue
            
            # Sentiment and topics are now included in article_data
            articles.append(article_data)
        
        # Ensure we have at least some articles
        if not articles:
            return {
                "company": company_name,
                "articles": [],
                "comparative_sentiment_score": {},
                "final_sentiment_analysis": "No valid articles found for analysis.",
                "audio": ""
            }
        
        # Perform comparative analysis
        comparative_analysis = perform_comparative_analysis(articles)
        
        # Format the results
        result = {
            "company": company_name,
            "articles": [
                {
                    "title": article["title"],
                    "summary": article["summary"],
                    "sentiment": article["sentiment"]["category"],
                    "topics": article["topics"],
                    "url": article["url"],
                    "source": article["source"],
                    "date": article["date"]
                }
                for article in articles
            ],
            "comparative_sentiment_score": comparative_analysis,
            "final_sentiment_analysis": comparative_analysis["final_sentiment_analysis"],
            "audio": ""  # Will be filled later
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing company news: {e}")
        return {
            "company": company_name,
            "articles": [],
            "comparative_sentiment_score": {},
            "final_sentiment_analysis": f"Error processing news: {str(e)}",
            "audio": ""
        }

def generate_summary_for_tts(analysis_result: Dict[str, Any]) -> str:
    """
    Generate a summary text for TTS conversion.
    
    Args:
        analysis_result: Analysis result dictionary
        
    Returns:
        Summary text for TTS
    """
    company = analysis_result["company"]
    
    # Count sentiments
    sentiment_counts = analysis_result["comparative_sentiment_score"].get("sentiment_distribution", {})
    total = sum(sentiment_counts.values())
    
    if total == 0:
        return f"{company} के बारे में कोई समाचार नहीं मिला।"
    
    # Calculate percentages
    positive_pct = int((sentiment_counts.get("Positive", 0) / total) * 100) if total > 0 else 0
    negative_pct = int((sentiment_counts.get("Negative", 0) / total) * 100) if total > 0 else 0
    neutral_pct = int((sentiment_counts.get("Neutral", 0) / total) * 100) if total > 0 else 0
    
    # Get common topics
    common_topics = analysis_result["comparative_sentiment_score"].get("topic_overlap", {}).get("common_topics", [])
    topics_text = ", ".join(common_topics[:3]) if common_topics else "कोई सामान्य विषय नहीं"
    
    # Create summary
    summary = f"{company} के बारे में समाचार विश्लेषण। "
    summary += f"कुल {total} समाचार लेखों का विश्लेषण किया गया। "
    summary += f"{positive_pct}% सकारात्मक, {negative_pct}% नकारात्मक, और {neutral_pct}% तटस्थ समाचार मिले। "
    
    if common_topics:
        summary += f"मुख्य विषय हैं: {topics_text}। "
    
    summary += analysis_result["final_sentiment_analysis"]
    
    # Translate to Hindi (in a real implementation, use a proper translation API)
    hindi_summary = translate_to_hindi(summary)
    
    return hindi_summary
