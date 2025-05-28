import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import uuid
from textblob import TextBlob
import ta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Market Prediction API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URL)
db = client.crypto_predictions

# API Keys from environment
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
GRAPH_API_KEY = os.environ.get('GRAPH_API_KEY')

class CryptoDataFetcher:
    def __init__(self):
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        self.news_api_base = "https://newsapi.org/v2/everything"
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = 1800  # 30 minutes
    
    async def fetch_crypto_data(self, symbol: str) -> Dict:
        """Fetch crypto data from Alpha Vantage with caching"""
        cache_key = f"crypto_{symbol}"
        current_time = datetime.now().timestamp()
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_duration:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
        
        try:
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': symbol,
                'market': 'USD',
                'apikey': ALPHA_VANTAGE_KEY
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.alpha_vantage_base, params=params) as response:
                    data = await response.json()
                    
                    # Check if we hit rate limit
                    if 'Information' in data and 'rate limit' in data['Information']:
                        logger.warning(f"Alpha Vantage rate limit hit: {data['Information']}")
                        # Try to use cached data even if expired
                        if cache_key in self.cache:
                            cached_data, _ = self.cache[cache_key]
                            logger.info(f"Using expired cached data for {symbol}")
                            return cached_data
                        return {"error": "Rate limit exceeded"}
                    
                    # Cache successful response
                    if 'Time Series (Digital Currency Daily)' in data:
                        self.cache[cache_key] = (data, current_time)
                    
                    return data
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            # Try to use cached data even if expired
            if cache_key in self.cache:
                cached_data, _ = self.cache[cache_key]
                logger.info(f"Using expired cached data for {symbol} due to error")
                return cached_data
            return {}
    
    async def fetch_crypto_news(self, symbol: str) -> List[Dict]:
        """Fetch crypto news for sentiment analysis with caching"""
        cache_key = f"news_{symbol}"
        current_time = datetime.now().timestamp()
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_duration:
                logger.info(f"Using cached news for {symbol}")
                return cached_data
        
        try:
            params = {
                'q': f'{symbol} cryptocurrency',
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 20,
                'apiKey': NEWS_API_KEY
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.news_api_base, params=params) as response:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    # Cache successful response
                    if articles:
                        self.cache[cache_key] = (articles, current_time)
                    
                    return articles
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            # Try to use cached data even if expired
            if cache_key in self.cache:
                cached_data, _ = self.cache[cache_key]
                logger.info(f"Using expired cached news for {symbol} due to error")
                return cached_data
            return []

class CryptoPredictionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for prediction"""
        # Price-based indicators
        df['SMA_7'] = ta.trend.sma_indicator(df['close'], window=7)
        df['SMA_21'] = ta.trend.sma_indicator(df['close'], window=21)
        df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['close'])
        
        # Volatility indicators
        df['BB_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # Volume indicators
        if 'volume' in df.columns:
            df['Volume_SMA'] = ta.trend.sma_indicator(df['volume'], window=20)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        feature_columns = [
            'SMA_7', 'SMA_21', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 
            'BB_width', 'Volume_SMA', 'open', 'high', 'low', 'volume'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            return np.array([])
        
        features = df[available_features].values
        return features
    
    def train_and_predict(self, df: pd.DataFrame) -> Dict:
        """Train model and make prediction"""
        try:
            if len(df) < 30:  # Need sufficient data
                return {"prediction": None, "confidence": 0, "error": "Insufficient data"}
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Prepare features and target
            features = self.prepare_features(df)
            
            if features.size == 0:
                return {"prediction": None, "confidence": 0, "error": "No valid features"}
            
            # Remove NaN values
            df_clean = df.dropna()
            if len(df_clean) < 20:
                return {"prediction": None, "confidence": 0, "error": "Insufficient clean data"}
            
            features_clean = self.prepare_features(df_clean)
            target = df_clean['close'].values
            
            # Use last N days for training
            train_size = min(len(df_clean) - 1, 100)
            X_train = features_clean[:train_size]
            y_train = target[1:train_size + 1]  # Next day's price
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make prediction for next day
            last_features = features_clean[-1:] 
            last_features_scaled = self.scaler.transform(last_features)
            prediction = self.model.predict(last_features_scaled)[0]
            
            # Calculate confidence based on recent prediction accuracy
            recent_predictions = self.model.predict(X_train_scaled[-10:])
            recent_actual = y_train[-10:]
            mape = np.mean(np.abs((recent_actual - recent_predictions) / recent_actual)) * 100
            confidence = max(0, 100 - mape)
            
            current_price = df_clean['close'].iloc[-1]
            price_change = ((prediction - current_price) / current_price) * 100
            
            return {
                "prediction": float(prediction),
                "current_price": float(current_price),
                "predicted_change": float(price_change),
                "confidence": float(confidence),
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {"prediction": None, "confidence": 0, "error": str(e)}

class SentimentAnalyzer:
    @staticmethod
    def analyze_news_sentiment(articles: List[Dict]) -> Dict:
        """Analyze sentiment of news articles"""
        if not articles:
            return {"sentiment_score": 0, "sentiment_label": "neutral", "article_count": 0}
        
        sentiments = []
        for article in articles[:10]:  # Analyze top 10 articles
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if text.strip():
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
        
        if not sentiments:
            return {"sentiment_score": 0, "sentiment_label": "neutral", "article_count": 0}
        
        avg_sentiment = np.mean(sentiments)
        
        # Classify sentiment
        if avg_sentiment > 0.1:
            label = "positive"
        elif avg_sentiment < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "sentiment_score": float(avg_sentiment),
            "sentiment_label": label,
            "article_count": len(sentiments)
        }

# Initialize components
data_fetcher = CryptoDataFetcher()
prediction_model = CryptoPredictionModel()
sentiment_analyzer = SentimentAnalyzer()

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/crypto/{symbol}")
async def get_crypto_data(symbol: str):
    """Get current crypto data and basic info"""
    try:
        logger.info(f"Fetching crypto data for {symbol}")
        logger.info(f"API Key available: {bool(ALPHA_VANTAGE_KEY)}")
        
        # Fetch current data
        data = await data_fetcher.fetch_crypto_data(symbol.upper())
        logger.info(f"Received data keys: {list(data.keys()) if data else 'No data'}")
        
        if not data or 'Time Series (Digital Currency Daily)' not in data:
            logger.error(f"Invalid data structure: {data}")
            raise HTTPException(status_code=404, detail="Crypto data not found")
        
        time_series = data['Time Series (Digital Currency Daily)']
        latest_date = max(time_series.keys())
        latest_data = time_series[latest_date]
        logger.info(f"Latest data keys: {list(latest_data.keys())}")
        
        result = {
            "symbol": symbol.upper(),
            "current_price": float(latest_data['4. close']),
            "open": float(latest_data['1. open']),
            "high": float(latest_data['2. high']),
            "low": float(latest_data['3. low']),
            "volume": float(latest_data['5. volume']),
            "last_updated": latest_date
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting crypto data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/{symbol}")
async def predict_crypto(symbol: str):
    """Get crypto prediction with sentiment analysis"""
    try:
        # Fetch historical data
        crypto_data = await data_fetcher.fetch_crypto_data(symbol.upper())
        
        if not crypto_data or 'Time Series (Digital Currency Daily)' not in crypto_data:
            raise HTTPException(status_code=404, detail="Unable to fetch crypto data")
        
        # Convert to DataFrame
        time_series = crypto_data['Time Series (Digital Currency Daily)']
        df_data = []
        
        for date, values in time_series.items():
            df_data.append({
                'date': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': float(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Get prediction
        prediction_result = prediction_model.train_and_predict(df)
        
        # Fetch news sentiment
        news_articles = await data_fetcher.fetch_crypto_news(symbol)
        sentiment_result = sentiment_analyzer.analyze_news_sentiment(news_articles)
        
        # Combine results
        result = {
            "symbol": symbol.upper(),
            "prediction": prediction_result,
            "sentiment": sentiment_result,
            "historical_data": df_data[-30:],  # Last 30 days
            "prediction_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat()
        }
        
        # Store prediction in database
        result_copy = result.copy()
        result_copy["_id"] = result["prediction_id"]  # Use UUID as _id
        db.predictions.insert_one(result_copy)
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting crypto: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/history/{symbol}")
async def get_prediction_history(symbol: str, limit: int = 10):
    """Get historical predictions for a symbol"""
    try:
        predictions = list(db.predictions.find(
            {"symbol": symbol.upper()},
            {"_id": 0}
        ).sort("created_at", -1).limit(limit))
        
        return {"symbol": symbol.upper(), "predictions": predictions}
    
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supported-cryptos")
async def get_supported_cryptos():
    """Get list of supported cryptocurrencies"""
    return {
        "supported": [
            {"symbol": "BTC", "name": "Bitcoin"},
            {"symbol": "ETH", "name": "Ethereum"},
            {"symbol": "ADA", "name": "Cardano"},
            {"symbol": "DOT", "name": "Polkadot"},
            {"symbol": "LINK", "name": "Chainlink"},
            {"symbol": "LTC", "name": "Litecoin"},
            {"symbol": "XRP", "name": "Ripple"},
            {"symbol": "BCH", "name": "Bitcoin Cash"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)