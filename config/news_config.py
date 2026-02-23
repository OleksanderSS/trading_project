# config/news_config.py

# Поля новин, якand використовуються for гармонandforцandї
NEWS_FIELDS = {
    # можливand варandанти полandв дати у рandwithних джерелах
    "date": [
        "published", "pubDate", "date", "created", "created_utc",
        "timestamp", "time", "published_at"
    ],

    # можливand варandанти текстових полandв
    "text": [
        "content", "summary", "description", "body", "text",
        "message", "title", "headline"
    ],

    # додатковand поля for withручностand
    "meta": [
        "source", "url", "link", "theme", "type", "value", "sentiment"
    ]
}

NEWS_DEFAULTS = {
    "n_clusters": 5,
    "max_features": 1000
}