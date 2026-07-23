# Real-Time AI Trading System

An AI-powered trading system that generates real-time **BUY**, **HOLD**, and **SELL** signals using Machine Learning. The application provides live market insights through FastAPI, WebSockets, and an interactive dashboard.

## Features

- Real-time trading signals
- Machine Learning ensemble model
- Technical indicator analysis (RSI, SMA, EMA)
- Walk-forward validation
- Strategy backtesting
- FastAPI REST API
- WebSocket live updates
- Interactive dashboard with TradingView charts

## 🛠️ Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Optuna
- yfinance
- HTML
- CSS
- JavaScript


## Installation

Clone the repository:

```bash
git clone https://github.com/shreeharsha0125-dotcom/real-time-ai-trading-system.git
cd real-time-ai-trading-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000
```

## Live Demo

**Live Application:** https://real-time-ai-trading-system.onrender.com

## GitHub Repository

**Repository:** https://github.com/shreeharsha0125-dotcom/real-time-ai-trading-system

##  Dashboard

The dashboard displays:

- BUY / HOLD / SELL Signals
- Confidence Score
- RSI
- Current Price
- Sharpe Ratio
- CAGR
- Maximum Drawdown
- Live TradingView Charts

##  License

This project is licensed under the MIT License.
