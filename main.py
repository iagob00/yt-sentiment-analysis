import os
import csv
import re
import nltk
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("punkt")
load_dotenv()
# Iniciar YouTube API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

analyzer = SentimentIntensityAnalyzer()

# Lexicon em portugues para o VADER
portuguese_sentiments = {
    "bom": 2.0, "ótimo": 3.0, "excelente": 3.5,
    "ruim": -2.0, "péssimo": -3.5, "horrível": -4.0,
    "feliz": 2.5, "triste": -2.5, "raiva": -3.0, "amor": 3.0,
    "incrível": 3.0, "terrível": -3.0, "perfeito": 3.5,
    "odio": -3.5, "medo": -2.5, "nojo": -3.0,
    "top": 2.5, "show": 2.0, "daora": 2.3, "chato": -2.0, "bosta": -3.5,
    "zoado": -2.8, "foda": 3.0, "maravilhoso": 4.0, "tmj" : 2.0, "melhor" : 2.0, "boa": 2.0,
    "hahaha" : 2.0, "kkkk" : 2.0, "parabens" : 3.0, "vagabundo" : -3.0, "merda" : -3.0
}
analyzer.lexicon.update(portuguese_sentiments)


def clean_comment(comment):
    """Remove emojis e caracteres spam."""
    comment = re.sub(r'[^a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ!?., ]', '', comment)
    return comment.strip()


def get_youtube_comments(video_id, max_comments=100):
    """Busca os comentários video."""
    comments = []
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=max_comments, textFormat="plainText"
    )
    response = request.execute()

    for item in response.get("items", []):
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        clean_text = clean_comment(text)
        if clean_text:
            comments.append(clean_text)

    return comments


def analyze_sentiment(comments):
    """Executa a analise de sentimentos."""
    results = []
    for comment in comments:
        sentiment_score = analyzer.polarity_scores(comment)["compound"]
        sentiment = "positive" if sentiment_score > 0.05 else "negative" if sentiment_score < -0.05 else "neutral"
        results.append((comment, sentiment, sentiment_score))
    return results


def save_results_to_csv(results, filename="youtube_sentiment.csv"):
    """Salva os resultados em um CSV."""
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Comment", "Sentiment", "Score"])
        writer.writerows(results)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    video_id = input("Enter YouTube Video ID: ")
    comments = get_youtube_comments(video_id)
    if not comments:
        print("No comments found!")
    else:
        analyzed_results = analyze_sentiment(comments)
        save_results_to_csv(analyzed_results)
        print("Sentiment analysis complete!")
