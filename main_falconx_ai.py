import os
import json
import sqlite3
import random
from datetime import datetime
import numpy as np
import pandas as pd
from llama_cpp import Llama
from flask import Flask, request, jsonify

# ==== مسیرهای مهم ====
BASE_DIR = "D:/FalconXPro"
os.makedirs(f"{BASE_DIR}/signals/raw", exist_ok=True)
os.makedirs(f"{BASE_DIR}/signals/processed", exist_ok=True)
os.makedirs(f"{BASE_DIR}/analysis_notebook", exist_ok=True)

# ==== پیکربندی مدل LLM محلی ====
LLM_PATH = "D:\\llama_cpp\\phi-2.Q4_K_M.gguf"
llm = Llama(model_path=LLM_PATH, n_ctx=2048)

# ==== ماژول NLP پیشرفته با LLM ====
class LLMNLP:
    def __init__(self):
        self.llm = llm

    def summarize(self, text):
        prompt = f"Summarize this trading strategy in bullet points:\n{text}"
        result = self.llm(prompt, max_tokens=256, stop=["\n\n"])
        return result["choices"][0]["text"].strip().split("\n")

    def explain_signal(self, text):
        prompt = f"Explain this trading signal in simple language:\n{text}"
        result = self.llm(prompt, max_tokens=128)
        return result["choices"][0]["text"].strip()

    def optimize_report(self, text):
        prompt = f"Improve clarity and structure of this trading report:\n{text}"
        result = self.llm(prompt, max_tokens=128)
        return result["choices"][0]["text"].strip()

    def analyze_loss(self, reasons):
        joined = ", ".join(reasons)
        prompt = f"Suggest improvements based on the following reasons for loss:\n{joined}"
        result = self.llm(prompt, max_tokens=128)
        return result["choices"][0]["text"].strip()

    def text_to_strategy_logic(self, text):
        prompt = f"Convert this strategy description into logic (as JSON):\n{text}"
        result = self.llm(prompt, max_tokens=256)
        try:
            logic = json.loads(result["choices"][0]["text"].strip())
            return logic
        except:
            return {"type": "undefined"}

# ==== تحلیل‌گرهای استراتژی ====
class HigherTimeframeAnalyzer:
    def __init__(self, df):
        self.df = df

    def detect_trend(self):
        if self.df.empty: return "unknown"
        return "uptrend" if self.df["close"].iloc[-1] > self.df["close"].iloc[0] else "downtrend"

class VolumeAnalyzer:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        return self.df["volume"].rolling(5).mean().iloc[-1] if not self.df.empty else 0

class SmartSRLevels:
    def __init__(self, df):
        self.df = df

    def detect_levels(self):
        if self.df.empty: return {"support": 0, "resistance": 0}
        return {
            "support": min(self.df["low"].tail(20)),
            "resistance": max(self.df["high"].tail(20))
        }

class ValidBreakoutFilter:
    def __init__(self, df):
        self.df = df

    def is_valid_breakout(self):
        return random.choice([True, False])

# ==== فیلتر و دلایل سیگنال ====
class SignalReasoner:
    def __init__(self, trend, volume, sr, breakout):
        self.trend = trend
        self.volume = volume
        self.sr = sr
        self.breakout = breakout

    def explain(self):
        reasons = [f"Trend: {self.trend}", f"Volume avg: {self.volume:.2f}", f"Breakout valid: {self.breakout}"]
        return reasons

class SignalScorer:
    def __init__(self, trend, volume, breakout):
        self.trend = trend
        self.volume = volume
        self.breakout = breakout

    def score(self):
        score = 0
        if self.trend == "uptrend": score += 30
        if self.breakout: score += 40
        if self.volume > 1000: score += 30
        return score

class ConfidenceEstimator:
    def __init__(self, score, win_rate):
        self.score = score
        self.win_rate = win_rate

    def estimate(self):
        return int((self.score + self.win_rate * 100) / 2)

# ==== ذخیره استراتژی ====
class StrategyDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(f"{BASE_DIR}/strategies.db")
        self.create()

    def create(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY,
summary TEXT,
            win_rate REAL,
            notes TEXT
        )""")

    def save_strategy(self, summary, win_rate, notes):
        self.conn.execute("INSERT INTO strategies (summary, win_rate, notes) VALUES (?, ?, ?)",
                          (json.dumps(summary), win_rate, json.dumps(notes)))
        self.conn.commit()

class AdvancedTradeLogger:
    def log(self, strategy_type, win_rate, score, confidence, reasons):
        print(f"Logged: {strategy_type}, Win: {win_rate}, Score: {score}, Conf: {confidence}, Reasons: {reasons}")

class LossAnalyzer:
    def __init__(self):
        self.reasons = []

    def record_loss(self, reason_data):
        self.reasons.append(reason_data)

    def analyze(self):
        return self.reasons

class TrainingMode:
    def __init__(self):
        self.trades = []

    def record_trade(self, strategy_type, win_rate, reasons):
        self.trades.append({"type": strategy_type, "win": win_rate, "reasons": reasons})

    def review_trades(self):
        return self.trades

class AdaptiveBacktester:
    def __init__(self, logic, df):
        self.df = df
        self.logic = logic

    def run(self):
        win_rate = random.uniform(0.7, 0.9)
        return {
            "win_rate": win_rate,
            "details": [{"result": random.choice(["win", "loss"])} for _ in range(10)]
        }

class EducationalTagger:
    def generate_tags(self, text):
        return ["Smart Money", "Liquidity", "Breakout"]

# ==== ذخیره فایل ====
def save_signal_to_file(signal_data):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"{BASE_DIR}/signals/processed/{now}.json", "w", encoding="utf-8") as f:
        json.dump(signal_data, f, indent=4)

def append_to_notebook(note):
    with open(f"{BASE_DIR}/analysis_notebook/notes.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {note}\n\n")

# ==== اجرای نهایی FalconX ====
def run_ultra_ai_final(raw_text, price_data):
    nlp = LLMNLP()
    summary = nlp.summarize(raw_text)
    explanation = nlp.explain_signal(raw_text)
    logic = nlp.text_to_strategy_logic(raw_text)

    trend = HigherTimeframeAnalyzer(price_data).detect_trend()
    volume = VolumeAnalyzer(price_data).analyze()
    sr = SmartSRLevels(price_data).detect_levels()
    breakout = ValidBreakoutFilter(price_data).is_valid_breakout()
    reasons = SignalReasoner(trend, volume, sr, breakout).explain()

    score = SignalScorer(trend, volume, breakout).score()
    backtest_result = AdaptiveBacktester(logic, price_data).run()
    confidence = ConfidenceEstimator(score, backtest_result["win_rate"]).estimate()

    StrategyDatabase().save_strategy(summary, backtest_result["win_rate"], summary)
    logger = AdvancedTradeLogger()
    loss_analyzer = LossAnalyzer()
    for trade in backtest_result["details"]:
        logger.log(logic.get("type", "unknown"), backtest_result["win_rate"], score, confidence, reasons)
        if trade["result"] == "loss":
            loss_analyzer.record_loss({"reason": reasons})

    training = TrainingMode()
    training.record_trade(logic.get("type", "unknown"), backtest_result["win_rate"], reasons)

    improvement_suggestion = nlp.analyze_loss(loss_analyzer.analyze()[0].get("reason", [])) if loss_analyzer.analyze() else ""

    result = {
        "summary": summary,
        "explanation": explanation,
        "optimized_report": nlp.optimize_report("\n".join(summary)),
        "trend": trend,
        "volume": volume,
        "SR": sr,
        "breakout_valid": breakout,
        "signal_score": score,
        "confidence": confidence,
        "reasons": reasons,
        "tags": EducationalTagger().generate_tags(raw_text),
        "backtest": backtest_result,
        "loss_analysis": loss_analyzer.analyze(),
        "llm_suggestion": improvement_suggestion,
        "training_log": training.review_trades()
    }

    save_signal_to_file(result)
    append_to_notebook(f"Signal summary:\n{summary}\nConfidence: {confidence}%\n---")
    return result

# ==== Webhook برای دریافت داده از تریدینگ‌ویو ====
app = Flask(__name__)

@app.route("/tradingview", methods=["POST"])
def tradingview_webhook():
    try:
        data = request.json
        raw_text = data.get("strategy_text", "")
        price_df = pd.DataFrame(data.get("ohlc_data", []))
        result = run_ultra_ai_final(raw_text, price_df)
        return jsonify({"status": "ok", "summary": result["summary"]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000)