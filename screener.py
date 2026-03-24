"""
グローバル中期投資スクリーニング 全27戦略（A-1〜I-4）
参照: screening_thresholds.txt
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
from datetime import datetime

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
TODAY        = datetime.today().strftime("%Y-%m-%d")
RUN_DATETIME = datetime.today().strftime("%Y-%m-%d_%H%M")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "results", RUN_DATETIME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# グローバル銘柄ユニバース（グロース株重視）
# ─────────────────────────────────────────
STOCK_UNIVERSE = [
    # ── 米国 メガキャップ・テック ──
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","TSM",
    # ── 米国 グロース・ソフトウェア ──
    "CRM","ADBE","NOW","SNOW","DDOG","ZS","CRWD","NET","MDB",
    "PLTR","ARM","ANET","FTNT","PANW","HUBS","RBLX","ABNB","UBER",
    # ── 米国 半導体 ──
    "AMD","INTC","QCOM","MU","MRVL","KLAC","LRCX","AMAT","MPWR","ASML",
    # ── 米国 フィンテック・Eコマース ──
    "SHOP","MELI","SE","COIN","PYPL","SOFI",
    # ── 米国 ヘルスケア・バイオ ──
    "LLY","NVO","MRNA","REGN","VRTX","ISRG","IDXX",
    # ── 米国 消費・小売 ──
    "COST","WMT","LULU","CAVA",
    # ── 米国 ディフェンシブ・配当 ──
    "JNJ","PG","KO","ABBV",
    # ── グローバル（欧州・アジア・新興国）──
    "SAP","INFY","BABA","PDD","GRAB",
    # ── 日本 大型・指数構成 ──
    "7203.T","6758.T","8306.T","9432.T","6861.T","4063.T","8035.T","9984.T",
    "6954.T","7974.T","4502.T","8316.T","6902.T","9433.T","4519.T","6501.T",
    "7751.T","8411.T","4661.T","8058.T","7741.T","6367.T","8766.T","9020.T",
    "7733.T","4578.T","8802.T","4543.T","6301.T","4568.T","4911.T","2802.T",
    "6762.T","6857.T","4751.T","3659.T","6594.T","7716.T","4755.T","4523.T",
    # ── 日本 中型・成長株 ──
    "6920.T","4385.T","4689.T","6098.T","4484.T","9270.T","4478.T","4433.T",
    "3923.T","4448.T","4371.T","3769.T","4532.T","6532.T","3064.T","9843.T",
    "3086.T","8267.T","9983.T","3048.T","2782.T","7453.T","3092.T","4307.T",
]

STOCK_UNIVERSE = list(dict.fromkeys(
    [t for t in STOCK_UNIVERSE
     if not t.endswith(".T") or (len(t.replace(".T","")) == 4 and t.replace(".T","").isdigit())]
))
print(f"対象銘柄数: {len(STOCK_UNIVERSE)}")

# ─────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────
def fetch_stock_data(tickers: list, period: str = "2y") -> dict:
    print(f"\n株価データ取得中... ({len(tickers)}銘柄)")
    data = {}
    for i in range(0, len(tickers), 10):
        batch = tickers[i:i+10]
        print(f"  {i+1}〜{min(i+10, len(tickers))}銘柄目...")
        for ticker in batch:
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if df is not None and len(df) >= 60:
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    data[ticker] = df
            except Exception:
                pass
        time.sleep(0.3)
    print(f"取得成功: {len(data)}銘柄")
    return data

def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

# ─────────────────────────────────────────
# テクニカル指標計算
# ─────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    # SMA（グローバル標準: 20/50/200）
    df["SMA20"]  = ta.sma(close, 20)
    df["SMA50"]  = ta.sma(close, 50)
    df["SMA200"] = ta.sma(close, 200)

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"]     = macd.get("MACD_12_26_9",     pd.Series(dtype=float))
        df["MACD_sig"] = macd.get("MACDs_12_26_9",    pd.Series(dtype=float))
        df["MACD_hist"]= macd.get("MACDh_12_26_9",    pd.Series(dtype=float))

    # RSI
    df["RSI14"] = ta.rsi(close, 14)

    # Bollinger Bands
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        df["BB_upper"] = bb.get("BBU_20_2.0", pd.Series(dtype=float))
        df["BB_mid"]   = bb.get("BBM_20_2.0", pd.Series(dtype=float))
        df["BB_lower"] = bb.get("BBL_20_2.0", pd.Series(dtype=float))

    # ADX / DI
    adx = ta.adx(df["High"], df["Low"], close, length=14)
    if adx is not None and not adx.empty:
        df["ADX"]    = adx.get("ADX_14",  pd.Series(dtype=float))
        df["DI_pos"] = adx.get("DMP_14",  pd.Series(dtype=float))
        df["DI_neg"] = adx.get("DMN_14",  pd.Series(dtype=float))

    # 出来高移動平均
    df["VOL_MA20"] = volume.rolling(20).mean()
    df["VOL_MA50"] = volume.rolling(50).mean()

    # 52週高値・安値
    df["HIGH_52W"] = close.rolling(252).max().shift(1)
    df["LOW_52W"]  = close.rolling(252).min()

    # リターン
    df["RET_20D"]  = close.pct_change(20) * 100
    df["RET_60D"]  = close.pct_change(60) * 100
    df["RET_120D"] = close.pct_change(120) * 100

    # 200日SMAからの乖離率
    df["DEV_SMA200"] = (close - df["SMA200"]) / df["SMA200"] * 100

    # ストキャスティクス
    stoch = ta.stoch(df["High"], df["Low"], close, k=14, d=3)
    if stoch is not None and not stoch.empty:
        df["STOCH_K"] = stoch.get("STOCHk_14_3_3", pd.Series(dtype=float))
        df["STOCH_D"] = stoch.get("STOCHd_14_3_3", pd.Series(dtype=float))

    return df

def get_latest(df: pd.DataFrame):
    return df.iloc[-1]

# ─────────────────────────────────────────
# スクリーニング関数
# ─────────────────────────────────────────

# ── A: モメンタム・ブレイクアウト ──

def screen_A1(df: pd.DataFrame, info: dict) -> bool:
    """A-1: 52週高値ブレイク + 出来高急増"""
    try:
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA50) or r.Volume < r.VOL_MA50 * 2.0: return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        rev_grw = info.get("revenueGrowth", 0) or 0
        if rev_grw < 0.15: return False
        return True
    except Exception: return False

def screen_A2(df: pd.DataFrame, info: dict) -> bool:
    """A-2: 多重MA完全順列 + ADX強トレンド"""
    try:
        r = get_latest(df)
        if any(pd.isna([r.SMA20, r.SMA50, r.SMA200])): return False
        if not (r.Close > r.SMA20 > r.SMA50 > r.SMA200): return False
        if pd.isna(r.ADX) or r.ADX < 30: return False
        if pd.isna(r.DI_pos) or pd.isna(r.DI_neg) or r.DI_pos <= r.DI_neg: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 75): return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.3: return False
        rev_grw = info.get("revenueGrowth", 0) or 0
        if rev_grw < 0.10: return False
        return True
    except Exception: return False

def screen_A3(df: pd.DataFrame) -> bool:
    """A-3: 相対強度トップ + 新高値継続"""
    try:
        r = get_latest(df)
        if pd.isna(r.RET_60D) or r.RET_60D < 15: return False
        if pd.isna(r.RET_20D) or r.RET_20D < 5: return False
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        return True
    except Exception: return False

# ── B: グロース × クオリティ ──

def screen_B1(df: pd.DataFrame, info: dict) -> bool:
    """B-1: Rule of 40 + テクニカル上昇"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        margin  = (info.get("operatingMargins", 0) or 0) * 100
        gross   = (info.get("grossMargins", 0) or 0) * 100
        if rev_grw + margin < 40: return False
        if rev_grw < 20: return False
        if gross < 50: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_B2(df: pd.DataFrame, info: dict) -> bool:
    """B-2: EPS加速成長 + 粗利拡大"""
    try:
        r = get_latest(df)
        roe   = info.get("returnOnEquity", 0) or 0
        gross = (info.get("grossMargins", 0) or 0) * 100
        eps_q = info.get("epsCurrentYear", None)
        eps_p = info.get("epsPreviousYear", None)
        if gross < 40: return False
        if roe < 0.15: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        # MACD買いシグナル（15日以内）
        for i in range(min(15, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig:
                return True
        return False
    except Exception: return False

def screen_B3(df: pd.DataFrame, info: dict) -> bool:
    """B-3: 売上高加速成長 + FCFマージン改善"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        margin  = (info.get("operatingMargins", 0) or 0) * 100
        if rev_grw < 20: return False
        if margin < 0: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

# ── C: GARP ──

def screen_C1(df: pd.DataFrame, info: dict) -> bool:
    """C-1: PEGレシオ割安 + 成長加速"""
    try:
        r = get_latest(df)
        peg     = info.get("pegRatio", None)
        per     = info.get("trailingPE", None)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if peg is None or peg > 1.5 or peg <= 0: return False
        if per is None or per < 10 or per < 5: return False
        if rev_grw < 15: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or r.RSI14 < 45: return False
        return True
    except Exception: return False

def screen_C2(df: pd.DataFrame, info: dict) -> bool:
    """C-2: 低PSR + 高成長（グロースバリュー）"""
    try:
        r = get_latest(df)
        psr     = info.get("priceToSalesTrailing12Months", None)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        gross   = (info.get("grossMargins", 0) or 0) * 100
        if psr is None or psr > 5.0: return False
        if rev_grw < 20: return False
        if gross < 40: return False
        # BB下限から反発
        if any(pd.isna([r.BB_lower, r.BB_mid])): return False
        if r.Close < r.BB_lower: return False
        if pd.isna(r.RSI14) or r.RSI14 < 35: return False
        return True
    except Exception: return False

def screen_C3(df: pd.DataFrame, info: dict) -> bool:
    """C-3: EV/Revenue割安 + 収益改善"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        margin  = (info.get("operatingMargins", 0) or 0) * 100
        if rev_grw < 15: return False
        if margin < -5: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        # MACD買いシグナル（10日以内）
        for i in range(min(10, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig:
                return True
        return False
    except Exception: return False

# ── D: カタリスト × テクニカル ──

def screen_D1(df: pd.DataFrame, info: dict) -> bool:
    """D-1: EPSサプライズ + ブレイクアウト"""
    try:
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if rev_grw < 10: return False
        return True
    except Exception: return False

def screen_D2(df: pd.DataFrame, info: dict) -> bool:
    """D-2: ガイダンス上方修正 + 新高値"""
    try:
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.5: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if rev_grw < 15: return False
        return True
    except Exception: return False

def screen_D3(df: pd.DataFrame, info: dict) -> bool:
    """D-3: 自社株買い大規模 + モメンタム"""
    try:
        r = get_latest(df)
        payout = info.get("payoutRatio", 1) or 1
        roe    = info.get("returnOnEquity", 0) or 0
        if roe < 0.10: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        for i in range(min(5, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig:
                return True
        return False
    except Exception: return False

# ── E: 需給転換・逆張り ──

def screen_E1(df: pd.DataFrame, info: dict) -> bool:
    """E-1: 強気MACDダイバージェンス + ファンダ良好"""
    try:
        roe     = info.get("returnOnEquity", 0) or 0
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if roe < 0.12: return False
        if rev_grw < 10: return False
        hist = df["MACD_hist"].dropna()
        rsi  = df["RSI14"].dropna()
        if len(hist) < 20 or len(rsi) < 20: return False
        # 直近2安値を検索（簡易版）
        lows = []
        for i in range(5, min(60, len(hist)-5)):
            idx = -(i)
            if hist.iloc[idx] < hist.iloc[idx-1] and hist.iloc[idx] < hist.iloc[idx+1]:
                lows.append(idx)
            if len(lows) == 2: break
        if len(lows) < 2: return False
        i1, i2 = lows[0], lows[1]
        close = df["Close"]
        if close.iloc[i1] >= close.iloc[i2]: return False  # 株価は安値更新
        if hist.iloc[i1] <= hist.iloc[i2]: return False    # ヒストグラムは逆行
        return True
    except Exception: return False

def screen_E2(df: pd.DataFrame, info: dict) -> bool:
    """E-2: ショートスクイーズ予兆 + 底打ち"""
    try:
        r = get_latest(df)
        if pd.isna(r.RSI14) or r.RSI14 > 40: return False
        recent_low = df["Close"].iloc[-20:].min()
        if r.Close <= recent_low * 1.01: return False
        if pd.isna(r.BB_lower) or r.Close < r.BB_lower: return False
        return True
    except Exception: return False

# ── F: グローバルテーマ ──

def screen_F1(df: pd.DataFrame, info: dict) -> bool:
    """F-1: AI・データインフラ セクターリーダー"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if rev_grw < 20: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 75): return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        return True
    except Exception: return False

def screen_F2(df: pd.DataFrame, info: dict) -> bool:
    """F-2: サイバーセキュリティ 成長株"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        gross   = (info.get("grossMargins", 0) or 0) * 100
        if rev_grw < 15: return False
        if gross < 65: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        # 52週高値から-20%以内
        if not pd.isna(r.HIGH_52W) and r.HIGH_52W > 0:
            if r.Close < r.HIGH_52W * 0.80: return False
        return True
    except Exception: return False

def screen_F3(df: pd.DataFrame, info: dict) -> bool:
    """F-3: バイオテック・ヘルステック カタリスト"""
    try:
        r = get_latest(df)
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.VOL_MA50) or r.Volume < r.VOL_MA50 * 1.5: return False
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if rev_grw < 5: return False
        return True
    except Exception: return False

# ── G: CANSLIM グローバル版 ──

def screen_G1(df: pd.DataFrame, info: dict) -> bool:
    """G-1: CANSLIM グローバル完全版"""
    try:
        r = get_latest(df)
        roe     = info.get("returnOnEquity", 0) or 0
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if roe < 0.17: return False
        if rev_grw < 15: return False
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA50) or r.Volume < r.VOL_MA50 * 2.0: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception: return False

def screen_G2(df: pd.DataFrame, info: dict) -> bool:
    """G-2: CANSLIM 簡易版（グローバル実践向け）"""
    try:
        r = get_latest(df)
        roe     = info.get("returnOnEquity", 0) or 0
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if roe < 0.15: return False
        if rev_grw < 15: return False
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 5: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception: return False

# ── H: 複合高精度 ──

def screen_H1(df: pd.DataFrame, info: dict) -> bool:
    """H-1: グロース × モメンタム × クオリティ 三軸"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        gross   = (info.get("grossMargins", 0) or 0) * 100
        if rev_grw < 20: return False
        if gross < 50: return False
        if pd.isna(r.SMA50) or pd.isna(r.SMA200): return False
        if not (r.Close > r.SMA50 > r.SMA200): return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        if pd.isna(r.RET_60D) or r.RET_60D < 15: return False
        return True
    except Exception: return False

def screen_H2(df: pd.DataFrame, info: dict) -> bool:
    """H-2: EPSサプライズ × ブレイクアウト × 相対強度 三軸"""
    try:
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        if pd.isna(r.RET_20D) or r.RET_20D < 3: return False
        return True
    except Exception: return False

def screen_H3(df: pd.DataFrame, info: dict) -> bool:
    """H-3: テーマセクター強 × 成長加速 × テクニカル 三軸"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if rev_grw < 20: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        if pd.isna(r.ADX) or r.ADX < 20: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        return True
    except Exception: return False

def screen_H4(df: pd.DataFrame, info: dict) -> bool:
    """H-4: 安定高品質 ディフェンシブグロース"""
    try:
        r = get_latest(df)
        roe     = info.get("returnOnEquity", 0) or 0
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        if roe < 0.15: return False
        if rev_grw < 10: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.DEV_SMA200) or r.DEV_SMA200 > 15: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        returns = df["Close"].pct_change().iloc[-120:]
        vol = returns.std()
        if pd.isna(vol) or vol > 0.025: return False
        return True
    except Exception: return False

# ── I: 追加戦略 ──

def screen_I1(df: pd.DataFrame, info: dict) -> bool:
    """I-1: 高ROIC企業"""
    try:
        r = get_latest(df)
        roe    = info.get("returnOnEquity", 0) or 0
        margin = (info.get("operatingMargins", 0) or 0) * 100
        de     = info.get("debtToEquity", 999) or 999
        if roe < 0.15: return False
        if margin < 15: return False
        if de > 50: return False  # D/E <= 0.5（yfinanceは%表記）
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_I2(df: pd.DataFrame, info: dict) -> bool:
    """I-2: ターンアラウンド（業績回復）"""
    try:
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        margin  = (info.get("operatingMargins", 0) or 0) * 100
        if rev_grw < 5: return False
        if margin < 0: return False  # 黒字転換確認
        if pd.isna(r.LOW_52W): return False
        if r.Close < r.LOW_52W * 1.20: return False  # 安値から+20%以上
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        for i in range(min(15, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig:
                return True
        return False
    except Exception: return False

def screen_I3(df: pd.DataFrame, info: dict) -> bool:
    """I-3: 小型株プレミアム（割安成長）"""
    try:
        r = get_latest(df)
        mktcap  = info.get("marketCap", 0) or 0
        rev_grw = (info.get("revenueGrowth", 0) or 0) * 100
        margin  = (info.get("operatingMargins", 0) or 0) * 100
        # 時価総額5000万〜5億USD
        if not (5e7 <= mktcap <= 5e8): return False
        if rev_grw < 10: return False
        if margin < 5: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.5: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_I4(df: pd.DataFrame, info: dict) -> bool:
    """I-4: 高配当 + 成長"""
    try:
        r = get_latest(df)
        div_yield = info.get("dividendYield", 0) or 0
        payout    = info.get("payoutRatio", 0) or 0
        rev_grw   = (info.get("revenueGrowth", 0) or 0) * 100
        if div_yield < 0.03: return False
        if not (0.30 <= payout <= 0.60): return False
        if rev_grw < 5: return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        return True
    except Exception: return False

# ─────────────────────────────────────────
# 戦略一覧
# ─────────────────────────────────────────
def needs_info_check(name: str) -> bool:
    keywords = ["A-1","A-2","B-","C-","D-","E-1","F-","G-","H-","I-"]
    return any(k in name for k in keywords)

STRATEGIES = {
    "A-1_高値BK出来高":       lambda df, info: screen_A1(df, info),
    "A-2_MA順列ADX":          lambda df, info: screen_A2(df, info),
    "A-3_相対強度新高値":      lambda df, info: screen_A3(df),
    "B-1_RuleOf40":           lambda df, info: screen_B1(df, info),
    "B-2_EPS加速粗利拡大":     lambda df, info: screen_B2(df, info),
    "B-3_売上加速FCF改善":     lambda df, info: screen_B3(df, info),
    "C-1_PEG割安成長加速":     lambda df, info: screen_C1(df, info),
    "C-2_低PSR高成長":         lambda df, info: screen_C2(df, info),
    "C-3_EVRev割安収益改善":   lambda df, info: screen_C3(df, info),
    "D-1_EPSサプライズBK":     lambda df, info: screen_D1(df, info),
    "D-2_上方修正新高値":      lambda df, info: screen_D2(df, info),
    "D-3_自社株買モメンタム":  lambda df, info: screen_D3(df, info),
    "E-1_MACDダイバージェンス": lambda df, info: screen_E1(df, info),
    "E-2_ショートスクイーズ":  lambda df, info: screen_E2(df, info),
    "F-1_AIデータインフラ":    lambda df, info: screen_F1(df, info),
    "F-2_サイバーセキュリティ": lambda df, info: screen_F2(df, info),
    "F-3_バイオヘルステック":  lambda df, info: screen_F3(df, info),
    "G-1_CANSLIM完全版":       lambda df, info: screen_G1(df, info),
    "G-2_CANSLIM簡易版":       lambda df, info: screen_G2(df, info),
    "H-1_グロースモメンタム":  lambda df, info: screen_H1(df, info),
    "H-2_サプライズBK強度":    lambda df, info: screen_H2(df, info),
    "H-3_テーマ成長加速":      lambda df, info: screen_H3(df, info),
    "H-4_安定ディフェンシブ":  lambda df, info: screen_H4(df, info),
    "I-1_高ROIC":              lambda df, info: screen_I1(df, info),
    "I-2_ターンアラウンド":    lambda df, info: screen_I2(df, info),
    "I-3_小型株プレミアム":    lambda df, info: screen_I3(df, info),
    "I-4_高配当成長":          lambda df, info: screen_I4(df, info),
}

# ─────────────────────────────────────────
# メイン実行
# ─────────────────────────────────────────
def run_all_screens():
    print("\nスクリーニング実行中...")
    stock_data = fetch_stock_data(STOCK_UNIVERSE, period="2y")

    results = []
    for ticker, raw_df in stock_data.items():
        try:
            df = calc_indicators(raw_df)
            info = fetch_info(ticker)

            hit_strategies = []
            for name, fn in STRATEGIES.items():
                try:
                    if fn(df, info):
                        hit_strategies.append(name)
                except Exception:
                    pass

            if hit_strategies:
                r = df.iloc[-1]
                mktcap = info.get("marketCap", None)
                results.append({
                    "銘柄コード":    ticker,
                    "現在値":        round(float(r.Close), 1),
                    "RSI14":         round(float(r.RSI14), 1) if not pd.isna(r.RSI14) else None,
                    "PER":           round(info.get("trailingPE", None) or 0, 1) or None,
                    "PBR":           round(info.get("priceToBook", None) or 0, 2) or None,
                    "ROE(%)":        round((info.get("returnOnEquity", 0) or 0) * 100, 1),
                    "売上成長(%)":   round((info.get("revenueGrowth", 0) or 0) * 100, 1),
                    "粗利率(%)":     round((info.get("grossMargins", 0) or 0) * 100, 1),
                    "時価総額(B$)":  round(mktcap / 1e9, 1) if mktcap else None,
                    "マッチ戦略数":  len(hit_strategies),
                    "マッチ戦略":    " | ".join(hit_strategies),
                })
        except Exception:
            pass

    return results

def save_results(results: list):
    if not results:
        print("ヒット銘柄なし")
        return

    df_all = pd.DataFrame(results).sort_values("マッチ戦略数", ascending=False)

    # ── CSV出力 ──
    path_all = os.path.join(OUTPUT_DIR, "screening_all.csv")
    df_all.to_csv(path_all, index=False, encoding="utf-8-sig")
    print(f"[保存] 全銘柄統合: {path_all}")

    # 戦略別展開
    rows = []
    for _, r in df_all.iterrows():
        for s in r["マッチ戦略"].split(" | "):
            row = r.to_dict()
            row["戦略"] = s
            rows.append(row)
    df_by = pd.DataFrame(rows)
    path_by = os.path.join(OUTPUT_DIR, "screening_by_strategy.csv")
    df_by.to_csv(path_by, index=False, encoding="utf-8-sig")
    print(f"[保存] 戦略別:    {path_by}")

    # サマリー
    summary = []
    for name in STRATEGIES:
        hits = [r for r in results if name in r["マッチ戦略"]]
        summary.append({
            "戦略": name,
            "ヒット銘柄数": len(hits),
            "銘柄一覧": ",".join([r["銘柄コード"] for r in hits]),
        })
    df_sum = pd.DataFrame(summary)
    path_sum = os.path.join(OUTPUT_DIR, "screening_summary.csv")
    df_sum.to_csv(path_sum, index=False, encoding="utf-8-sig")
    print(f"[保存] サマリー:   {path_sum}")

    # ── TXTレポート ──
    path_txt = os.path.join(OUTPUT_DIR, "screening_report.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f" スクリーニング結果サマリー ({TODAY})\n")
        f.write("=" * 60 + "\n")
        f.write(f"スキャン銘柄数: {len(STOCK_UNIVERSE)}\n")
        f.write(f"ヒット銘柄数:   {len(results)}\n\n")
        f.write(df_sum.to_string(index=False) + "\n\n")
        f.write("【全ヒット銘柄】\n")
        cols = ["銘柄コード","現在値","RSI14","PER","PBR","ROE(%)","売上成長(%)","粗利率(%)","マッチ戦略数","マッチ戦略"]
        f.write(df_all[cols].to_string(index=False) + "\n")
    print(f"[保存] レポート:   {path_txt}")

    # ── コンソール出力 ──
    print("\n" + "=" * 60)
    print(f" スクリーニング結果サマリー ({TODAY})")
    print("=" * 60)
    print(f"スキャン銘柄数: {len(STOCK_UNIVERSE)}")
    print(f"ヒット銘柄数:   {len(results)}\n")
    print(df_sum.to_string(index=False))
    print("\n【全ヒット銘柄】")
    print(df_all[cols].to_string(index=False))

if __name__ == "__main__":
    results = run_all_screens()
    save_results(results)
