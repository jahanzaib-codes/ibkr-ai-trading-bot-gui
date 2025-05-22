# 📈 IBKR AI Trading Bot with GUI

A Python-based options trading bot with an interactive GUI, built for **Interactive Brokers (IBKR)**. Integrates real-time data, AI-driven signals, and risk-managed strategies with tiered exits and stop-losses.
<p align="center">
  <img src="https://github.com/jahanzaib-codes/ibkr-ai-trading-bot-gui/blob/main/Capture.PNG?raw=true" width="600"/>
</p>



## 🚀 Features

- ✅ **IBKR Real-Time Connectivity** via `ib_insync`
- 💡 **AI/ML Powered**: XGBoost-based price movement prediction
- 📊 **Entry Filters**: VWAP alignment, RSI range, MACD crossovers, volume spikes
- 📈 **Tiered Take-Profit**: 25%, 75%, 150% profit scaling
- 🛑 **Stop-Loss Management**: Configurable risk percentage
- 🧠 **Optional OpenAI Integration** (future-ready)
- 🖥️ **Tkinter GUI**: User-friendly interface with real-time log display
- 🔁 **Backtesting Module** with dynamic inputs

---

## 🖥️ GUI Preview

> 🎯 Real-time backtest without freezing the UI

- Set Initial Capital
- Select Tickers (comma-separated)
- Allocation % per trade
- Stop-loss %
- Click `Start Backtest` to simulate

---

## 🧠 AI Model

- Trained on historical data (XGBoost)
- Features:
  - VWAP
  - RSI
  - MACD/Signal
  - Average Volume

---

## ⚙️ Requirements

Install all required packages:
```bash
pip install -r requirements.txt
