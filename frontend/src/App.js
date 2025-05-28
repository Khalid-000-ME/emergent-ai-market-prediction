import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'https://emergent-ai-market-prediction.onrender.com';

function App() {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [cryptoData, setCryptoData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [supportedCryptos, setSupportedCryptos] = useState([]);

  // Fetch supported cryptocurrencies
  useEffect(() => {
    fetchSupportedCryptos();
  }, []);

  // Fetch data when crypto selection changes
  useEffect(() => {
    if (selectedCrypto) {
      fetchCryptoData();
      fetchPrediction();
      fetchPredictionHistory();
    }
  }, [selectedCrypto]);

  const fetchSupportedCryptos = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/supported-cryptos`);
      const data = await response.json();
      setSupportedCryptos(data.supported);
    } catch (err) {
      console.error('Error fetching supported cryptos:', err);
    }
  };

  const fetchCryptoData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE_URL}/api/crypto/${selectedCrypto}`);
      
      if (response.status === 429) {
        // Rate limit hit - try to use prediction history for current price
        const historyResponse = await fetch(`${API_BASE_URL}/api/predictions/history/${selectedCrypto}`);
        if (historyResponse.ok) {
          const historyData = await historyResponse.json();
          if (historyData.predictions && historyData.predictions.length > 0) {
            const latestPrediction = historyData.predictions[0];
            if (latestPrediction.prediction.current_price) {
              setCryptoData({
                symbol: selectedCrypto,
                current_price: latestPrediction.prediction.current_price,
                open: latestPrediction.prediction.current_price * 0.995, // Estimate
                high: latestPrediction.prediction.current_price * 1.002, // Estimate
                low: latestPrediction.prediction.current_price * 0.998, // Estimate
                volume: 1000, // Placeholder
                last_updated: latestPrediction.created_at.split('T')[0],
                cached: true,
                rate_limited: true
              });
              setError("Using cached data due to API rate limit. Upgrade API plan for real-time data.");
              return;
            }
          }
        }
        throw new Error('API rate limit exceeded. Please try again later.');
      }
      
      if (!response.ok) {
        throw new Error('Failed to fetch crypto data');
      }
      
      const data = await response.json();
      setCryptoData(data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching crypto data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchPrediction = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/predict/${selectedCrypto}`);
      
      if (response.status === 429) {
        // Rate limit hit - use existing prediction history
        const historyResponse = await fetch(`${API_BASE_URL}/api/predictions/history/${selectedCrypto}`);
        if (historyResponse.ok) {
          const historyData = await historyResponse.json();
          if (historyData.predictions && historyData.predictions.length > 0) {
            setPrediction(historyData.predictions[0]);
            setError("Showing latest cached prediction due to API rate limit.");
            return;
          }
        }
        throw new Error('API rate limit exceeded. Please try again later.');
      }
      
      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching prediction:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchPredictionHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/predictions/history/${selectedCrypto}`);
      const data = await response.json();
      setPredictionHistory(data.predictions || []);
    } catch (err) {
      console.error('Error fetching prediction history:', err);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(price);
  };

  const formatPercentage = (value) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600';
      case 'negative': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive': return '📈';
      case 'negative': return '📉';
      default: return '➖';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 70) return 'text-green-600';
    if (confidence >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getChangeColor = (change) => {
    return change >= 0 ? 'text-green-600' : 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
      {/* Header */}
      <header className="bg-black bg-opacity-20 backdrop-blur-lg border-b border-white border-opacity-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">₿</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">CryptoPredictAI</h1>
                <p className="text-purple-200 text-sm">AI-Powered Market Predictions</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <select
                value={selectedCrypto}
                onChange={(e) => setSelectedCrypto(e.target.value)}
                className="bg-white bg-opacity-10 backdrop-blur border border-white border-opacity-20 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                {supportedCryptos.map((crypto) => (
                  <option key={crypto.symbol} value={crypto.symbol} className="bg-gray-800">
                    {crypto.symbol} - {crypto.name}
                  </option>
                ))}
              </select>
              
              <button
                onClick={() => {
                  fetchCryptoData();
                  fetchPrediction();
                  fetchPredictionHistory();
                }}
                disabled={loading}
                className="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 disabled:opacity-50 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 transform hover:scale-105"
              >
                {loading ? 'Updating...' : 'Refresh'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="bg-red-500 bg-opacity-10 border border-red-500 border-opacity-50 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <span className="text-red-400 mr-2">⚠️</span>
              <span className="text-red-300">{error}</span>
            </div>
          </div>
        )}

        {/* Current Price Card */}
        {cryptoData && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-2 bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6 border border-white border-opacity-10">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">Current Price</h2>
                <span className="text-purple-300 text-sm">{cryptoData.last_updated}</span>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-gray-300 text-sm mb-1">Current</p>
                  <p className="text-2xl font-bold text-white">{formatPrice(cryptoData.current_price)}</p>
                </div>
                <div className="text-center">
                  <p className="text-gray-300 text-sm mb-1">Open</p>
                  <p className="text-lg font-semibold text-purple-300">{formatPrice(cryptoData.open)}</p>
                </div>
                <div className="text-center">
                  <p className="text-gray-300 text-sm mb-1">High</p>
                  <p className="text-lg font-semibold text-green-400">{formatPrice(cryptoData.high)}</p>
                </div>
                <div className="text-center">
                  <p className="text-gray-300 text-sm mb-1">Low</p>
                  <p className="text-lg font-semibold text-red-400">{formatPrice(cryptoData.low)}</p>
                </div>
              </div>
            </div>

            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6 border border-white border-opacity-10">
              <h3 className="text-lg font-semibold text-white mb-4">24h Volume</h3>
              <div className="text-center">
                <p className="text-2xl font-bold text-purple-300">
                  {new Intl.NumberFormat('en-US', {
                    notation: 'compact',
                    maximumFractionDigits: 1
                  }).format(cryptoData.volume)}
                </p>
                <p className="text-gray-400 text-sm mt-1">{selectedCrypto}</p>
              </div>
            </div>
          </div>
        )}

        {/* Prediction Card */}
        {prediction && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="bg-gradient-to-br from-purple-600 to-blue-600 bg-opacity-20 backdrop-blur-lg rounded-2xl p-6 border border-purple-500 border-opacity-30">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white">AI Prediction</h2>
                <span className="bg-purple-500 bg-opacity-30 text-purple-200 px-3 py-1 rounded-full text-sm">
                  Daily Forecast
                </span>
              </div>

              {prediction.prediction.prediction ? (
                <div className="space-y-4">
                  <div className="text-center">
                    <p className="text-gray-300 text-sm mb-2">Predicted Price (Tomorrow)</p>
                    <p className="text-3xl font-bold text-white mb-2">
                      {formatPrice(prediction.prediction.prediction)}
                    </p>
                    <p className={`text-lg font-semibold ${getChangeColor(prediction.prediction.predicted_change)}`}>
                      {formatPercentage(prediction.prediction.predicted_change)}
                    </p>
                  </div>

                  <div className="bg-black bg-opacity-20 rounded-lg p-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Confidence Score</span>
                      <span className={`font-bold ${getConfidenceColor(prediction.prediction.confidence)}`}>
                        {prediction.prediction.confidence.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${prediction.prediction.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-red-300">Unable to generate prediction</p>
                  <p className="text-gray-400 text-sm mt-2">{prediction.prediction.error}</p>
                </div>
              )}
            </div>

            {/* Sentiment Analysis */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6 border border-white border-opacity-10">
              <h3 className="text-xl font-semibold text-white mb-6">Market Sentiment</h3>
              
              <div className="text-center mb-6">
                <div className="text-4xl mb-2">
                  {getSentimentIcon(prediction.sentiment.sentiment_label)}
                </div>
                <p className={`text-xl font-semibold ${getSentimentColor(prediction.sentiment.sentiment_label)}`}>
                  {prediction.sentiment.sentiment_label.toUpperCase()}
                </p>
                <p className="text-gray-400 text-sm mt-1">
                  Based on {prediction.sentiment.article_count} news articles
                </p>
              </div>

              <div className="bg-black bg-opacity-20 rounded-lg p-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Sentiment Score</span>
                  <span className={`font-bold ${getSentimentColor(prediction.sentiment.sentiment_label)}`}>
                    {prediction.sentiment.sentiment_score.toFixed(3)}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      prediction.sentiment.sentiment_score >= 0 
                        ? 'bg-gradient-to-r from-green-500 to-green-400' 
                        : 'bg-gradient-to-r from-red-500 to-red-400'
                    }`}
                    style={{ 
                      width: `${Math.abs(prediction.sentiment.sentiment_score) * 100}%`,
                      marginLeft: prediction.sentiment.sentiment_score < 0 ? 'auto' : '0'
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Prediction History */}
        {predictionHistory.length > 0 && (
          <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6 border border-white border-opacity-10">
            <h3 className="text-xl font-semibold text-white mb-6">Recent Predictions</h3>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white border-opacity-10">
                    <th className="text-left text-gray-300 py-3">Date</th>
                    <th className="text-left text-gray-300 py-3">Predicted Price</th>
                    <th className="text-left text-gray-300 py-3">Change</th>
                    <th className="text-left text-gray-300 py-3">Confidence</th>
                    <th className="text-left text-gray-300 py-3">Sentiment</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionHistory.slice(0, 5).map((pred, index) => (
                    <tr key={index} className="border-b border-white border-opacity-5">
                      <td className="text-white py-3">
                        {new Date(pred.created_at).toLocaleDateString()}
                      </td>
                      <td className="text-white py-3">
                        {pred.prediction.prediction ? formatPrice(pred.prediction.prediction) : 'N/A'}
                      </td>
                      <td className={`py-3 ${getChangeColor(pred.prediction.predicted_change || 0)}`}>
                        {pred.prediction.predicted_change ? formatPercentage(pred.prediction.predicted_change) : 'N/A'}
                      </td>
                      <td className={`py-3 ${getConfidenceColor(pred.prediction.confidence || 0)}`}>
                        {pred.prediction.confidence ? `${pred.prediction.confidence.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className={`py-3 ${getSentimentColor(pred.sentiment.sentiment_label)}`}>
                        {pred.sentiment.sentiment_label || 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
            <p className="text-white mt-4">Loading AI predictions...</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;