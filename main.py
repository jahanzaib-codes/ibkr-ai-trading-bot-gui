from tkinter import *
from threading import Thread
import asyncio
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import time

# ─── GUI Setup ───────────────────────────────────────────────────────────────
root = Tk()
root.title("IBKR Backtest GUI")
root.geometry("700x600")

Label(root, text="Initial Capital").grid(row=0, column=0, sticky=W, padx=10)
initial_cap_entry = Entry(root)
initial_cap_entry.insert(0, "100000")
initial_cap_entry.grid(row=0, column=1)

Label(root, text="Tickers (comma-separated)").grid(row=1, column=0, sticky=W, padx=10)
tickers_entry = Entry(root)
tickers_entry.insert(0, "SPY,AAPL")
tickers_entry.grid(row=1, column=1)

Label(root, text="Allocation Per Trade (%)").grid(row=2, column=0, sticky=W, padx=10)
alloc_entry = Entry(root)
alloc_entry.insert(0, "10")
alloc_entry.grid(row=2, column=1)

Label(root, text="Stop-loss Threshold (%)").grid(row=3, column=0, sticky=W, padx=10)
sl_entry = Entry(root)
sl_entry.insert(0, "5")
sl_entry.grid(row=3, column=1)

output_text = Text(root, height=30, width=80, state='disabled', bg="#f0f0f0")
output_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

def log(message):
    output_text.configure(state='normal')
    output_text.insert(END, message + "\n")
    output_text.see(END)
    output_text.configure(state='disabled')

# ─── Backtest Logic ──────────────────────────────────────────────────────────
def run_backtest():
    asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        initial_cap = float(initial_cap_entry.get())
        tickers = [x.strip().upper() for x in tickers_entry.get().split(",")]
        alloc_pct = float(alloc_entry.get()) / 100
        sl_thresh = float(sl_entry.get()) / 100

        ib = IB()
        try:
            ib.connect('127.0.0.1', 7497, clientId=2)
        except Exception as e:
            log(f"❌ IB connection error: {e}")
            return

        def fetch_data(symbol, duration='14 D'):
            contract = Stock(symbol, 'SMART', 'USD', primaryExchange='ARCA')
            ib.qualifyContracts(contract)
            for _ in range(3):
                bars = ib.reqHistoricalData(contract, '', duration, '1 min', 'TRADES', True)
                if bars:
                    return util.df(bars).set_index('date')
                time.sleep(2)
            raise RuntimeError(f"Failed fetching {symbol}")

        def compute_indicators(df):
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            d = df['close'].diff()
            gain = d.clip(lower=0)
            loss = -d.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-5)
            df['rsi'] = 100 - (100 / (1 + rs))
            e12 = df['close'].ewm(span=12, adjust=False).mean()
            e26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = e12 - e26
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['avg_vol'] = df['volume'].rolling(20).mean()
            return df.dropna()

        model_file = 'xgb_odte_model.pkl'
        if os.path.exists(model_file):
            model = joblib.load(model_file)
        else:
            train_df = fetch_data('SPY', '90 D')
            train_df = compute_indicators(train_df)
            train_df['future'] = train_df['close'].shift(-10)
            train_df['label'] = (train_df['future'] > train_df['close']).astype(int)
            feats = train_df[['vwap', 'rsi', 'macd', 'signal', 'avg_vol']]
            labels = train_df['label']
            Xtr, Xte, ytr, yte = train_test_split(feats, labels, test_size=0.2, shuffle=False)
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
            model.fit(Xtr, ytr)
            joblib.dump(model, model_file)

        capital = initial_cap
        TP_TIERS = [1.25, 1.75, 2.50]
        DELTA = 0.5
        logs = []

        for sym in tickers:
            df = fetch_data(sym, '14 D')
            df = compute_indicators(df)
            log(f"\n🔁 Backtesting {sym}")

            position = None
            targets = []
            ti = 0

            for i in range(20, len(df)):
                row = df.iloc[i]
                p = row['close']
                t = row.name

                fake = (p > row['vwap'] and df.iloc[i - 1]['close'] > p and row['volume'] > 2 * row['avg_vol'])
                base = (p > row['vwap'] and 30 < row['rsi'] < 70 and row['macd'] > row['signal']
                        and row['volume'] > 1.5 * row['avg_vol'] and not fake)
                feat = np.array([[row['vwap'], row['rsi'], row['macd'], row['signal'], row['avg_vol']]])
                aiok = model.predict(feat)[0] == 1

                if position is None and base and aiok:
                    prem = p * 0.02
                    cnt = int((capital * alloc_pct) / (prem * 100))
                    position = {'entry_t': t, 'entry_p': p, 'cnt': cnt}
                    targets = [p * x for x in TP_TIERS]
                    ti = 0
                    log(f"[ENTRY] {t} {sym} @ {p:.2f} → {cnt} contracts")

                elif position:
                    move = p - position['entry_p']
                    pnl = move * DELTA * 100 * position['cnt']

                    if ti < len(targets) and p >= targets[ti]:
                        gain = (targets[ti] - position['entry_p']) * DELTA * 100 * (position['cnt'] // len(targets))
                        capital += gain
                        log(f"[TIER{ti + 1}] {t} {sym} PnL ${gain:.2f}")
                        ti += 1
                        if ti >= len(targets):
                            position = None

                    elif pnl <= -initial_cap * alloc_pct * sl_thresh:
                        capital += pnl
                        log(f"[SL] {t} {sym} PnL ${pnl:.2f}")
                        position = None

                    elif (row['rsi'] < df.iloc[i - 1]['rsi'] and
                          row['macd'] < df.iloc[i - 1]['macd'] and
                          row['volume'] < row['avg_vol']):
                        capital += pnl
                        log(f"[SmartExit] {t} {sym} PnL ${pnl:.2f}")
                        position = None

        log(f"\n💰 Final Capital: ${capital:.2f}")
        ib.disconnect()
    except Exception as e:
        log(f"❌ Error: {e}")

# ─── Button to Start Backtest ────────────────────────────────────────────────
Button(root, text="Start Backtest", command=lambda: Thread(target=run_backtest).start(), bg="#4CAF50", fg="white").grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
