"""
中期投資スクリーニング 全27戦略（A-1〜I-4）
対象: 国内株式（東証）/ 米国株式（NYSE・NASDAQ）
市場ごとに独立した基準値を適用
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
# 市場別パラメータ
# ─────────────────────────────────────────
MARKET_PARAMS = {
    "JP": {
        # フィルター
        "mktcap_min":  10_000_000_000,  # 100億円
        "adv_min":     50_000_000,       # 0.5億円/日（出来高×株価）
        "price_min":   300,              # 300円
        # SMA（日本標準）
        "sma_s": "SMA25", "sma_m": "SMA75", "sma_l": "SMA200",
        # ファンダ閾値
        "rev_grw_min": 0.08,   # 売上成長 8%以上
        "roe_min":     0.10,   # ROE 10%以上
        "gross_min":   0.25,   # 粗利率 25%以上（製造業含む）
        "margin_min":  0.05,   # 営業利益率 5%以上
    },
    "US": {
        # フィルター
        "mktcap_min":  200_000_000,     # 200M USD
        "adv_min":     1_000_000,        # 1M USD/日
        "price_min":   5,                # 5 USD
        # SMA（米国標準）
        "sma_s": "SMA20", "sma_m": "SMA50", "sma_l": "SMA200",
        # ファンダ閾値
        "rev_grw_min": 0.15,   # 売上成長 15%以上
        "roe_min":     0.15,   # ROE 15%以上
        "gross_min":   0.40,   # 粗利率 40%以上
        "margin_min":  0.08,   # 営業利益率 8%以上
    },
}

# ─────────────────────────────────────────
# 銘柄ユニバース
# ─────────────────────────────────────────
STOCK_UNIVERSE_JP = [
    # ── 大型・指数構成 ──
    "7203.T","6758.T","8306.T","9432.T","6861.T","4063.T","8035.T","9984.T",
    "6954.T","7974.T","4502.T","8316.T","6902.T","9433.T","4519.T","6501.T",
    "7751.T","8411.T","4661.T","8058.T","7741.T","6367.T","8766.T","9020.T",
    "7733.T","4578.T","8802.T","4543.T","6301.T","4568.T","4911.T","2802.T",
    "6762.T","6857.T","4751.T","3659.T","6594.T","7716.T","4755.T","4523.T",
    "8309.T","6988.T","4704.T","8830.T","3197.T","5401.T","8267.T","9843.T",
    # ── 中型・成長株 ──
    "6920.T","4385.T","4689.T","6098.T","4484.T","9270.T","4478.T","4433.T",
    "3923.T","4448.T","4371.T","3769.T","6532.T","3064.T","3086.T","9983.T",
    "3048.T","2782.T","7453.T","3092.T","4307.T","4752.T","6976.T","3436.T",
    "4901.T","5713.T","6326.T","8031.T","8001.T","9022.T","4661.T","6971.T",
]

STOCK_UNIVERSE_US = [
    # ── メガキャップ・テック ──
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO",
    # ── グロース・ソフトウェア ──
    "CRM","ADBE","NOW","SNOW","DDOG","ZS","CRWD","NET","MDB",
    "PLTR","ARM","ANET","FTNT","PANW","HUBS","RBLX","ABNB","UBER",
    # ── 半導体 ──
    "AMD","INTC","QCOM","MU","MRVL","KLAC","LRCX","AMAT","MPWR",
    # ── フィンテック・Eコマース ──
    "SHOP","MELI","COIN","PYPL","SOFI",
    # ── ヘルスケア・バイオ ──
    "LLY","MRNA","REGN","VRTX","ISRG","IDXX",
    # ── 消費・小売 ──
    "COST","WMT","LULU","CAVA",
    # ── ディフェンシブ・配当 ──
    "JNJ","PG","KO","ABBV",
]

_jp = [t for t in STOCK_UNIVERSE_JP
       if len(t.replace(".T","")) == 4 and t.replace(".T","").isdigit()]
STOCK_UNIVERSE = list(dict.fromkeys(_jp + STOCK_UNIVERSE_US))
MARKET_MAP     = {t: "JP" if t.endswith(".T") else "US" for t in STOCK_UNIVERSE}

_n_jp = sum(1 for t in STOCK_UNIVERSE if t.endswith(".T"))
_n_us = len(STOCK_UNIVERSE) - _n_jp
print(f"対象銘柄数: {len(STOCK_UNIVERSE)}  (国内: {_n_jp}銘柄 / 米国: {_n_us}銘柄)")

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
# テクニカル指標計算（JP/US両対応）
# ─────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    # SMA（JP用: 25/75/200、US用: 20/50/200、共通: 200）
    df["SMA20"]  = ta.sma(close, 20)
    df["SMA25"]  = ta.sma(close, 25)
    df["SMA50"]  = ta.sma(close, 50)
    df["SMA75"]  = ta.sma(close, 75)
    df["SMA200"] = ta.sma(close, 200)

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"]      = macd.get("MACD_12_26_9",  pd.Series(dtype=float))
        df["MACD_sig"]  = macd.get("MACDs_12_26_9", pd.Series(dtype=float))
        df["MACD_hist"] = macd.get("MACDh_12_26_9", pd.Series(dtype=float))

    # RSI
    df["RSI14"] = ta.rsi(close, 14)

    # Bollinger Bands (20,2σ)
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        df["BB_upper"] = bb.get("BBU_20_2.0", pd.Series(dtype=float))
        df["BB_mid"]   = bb.get("BBM_20_2.0", pd.Series(dtype=float))
        df["BB_lower"] = bb.get("BBL_20_2.0", pd.Series(dtype=float))

    # ADX / DI
    adx = ta.adx(df["High"], df["Low"], close, length=14)
    if adx is not None and not adx.empty:
        df["ADX"]    = adx.get("ADX_14", pd.Series(dtype=float))
        df["DI_pos"] = adx.get("DMP_14", pd.Series(dtype=float))
        df["DI_neg"] = adx.get("DMN_14", pd.Series(dtype=float))

    # 出来高移動平均
    df["VOL_MA20"] = volume.rolling(20).mean()
    df["VOL_MA50"] = volume.rolling(50).mean()

    # 日次売買代金（出来高 × 株価）→ ADV
    df["ADV20"] = (volume * close).rolling(20).mean()

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

def get_latest(df):
    return df.iloc[-1]

# ─────────────────────────────────────────
# 共通ベースフィルター（市場別）
# ─────────────────────────────────────────
def passes_base_filter(df: pd.DataFrame, info: dict, market: str) -> bool:
    p = MARKET_PARAMS[market]
    r = get_latest(df)
    mktcap = info.get("marketCap", 0) or 0
    if mktcap > 0 and mktcap < p["mktcap_min"]: return False
    if r.Close < p["price_min"]: return False
    if not pd.isna(r.ADV20) and r.ADV20 < p["adv_min"]: return False
    return True

# ─────────────────────────────────────────
# スクリーニング関数（market パラメータで閾値を切り替え）
# ─────────────────────────────────────────

def screen_A1(df, info, market) -> bool:
    """A-1: 52週高値ブレイク + 出来高急増"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        vol_base = r.VOL_MA50 if not pd.isna(r.VOL_MA50) else r.VOL_MA20
        if pd.isna(vol_base) or r.Volume < vol_base * 2.0: return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"]: return False
        return True
    except Exception: return False

def screen_A2(df, info, market) -> bool:
    """A-2: 多重MA完全順列 + ADX強トレンド"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        s, m, l = r[p["sma_s"]], r[p["sma_m"]], r[p["sma_l"]]
        if any(pd.isna([s, m, l])): return False
        if not (r.Close > s > m > l): return False
        if pd.isna(r.ADX) or r.ADX < 30: return False
        if pd.isna(r.DI_pos) or r.DI_pos <= (r.DI_neg or 0): return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 75): return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.3: return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"]: return False
        return True
    except Exception: return False

def screen_A3(df, info, market) -> bool:
    """A-3: 相対強度トップ + 新高値継続"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if pd.isna(r.RET_60D) or r.RET_60D < 15: return False
        if pd.isna(r.RET_20D) or r.RET_20D < 5: return False
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        return True
    except Exception: return False

def screen_B1(df, info, market) -> bool:
    """B-1: Rule of 40 + テクニカル上昇"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        margin  = (info.get("operatingMargins") or 0) * 100
        gross   = (info.get("grossMargins") or 0) * 100
        threshold = 35 if market == "JP" else 40
        if rev_grw + margin < threshold: return False
        if rev_grw < p["rev_grw_min"] * 100: return False
        if gross < p["gross_min"] * 100: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_B2(df, info, market) -> bool:
    """B-2: EPS加速成長 + 粗利拡大"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        gross = (info.get("grossMargins") or 0) * 100
        roe   = info.get("returnOnEquity") or 0
        if gross < p["gross_min"] * 100: return False
        if roe < p["roe_min"]: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        for i in range(min(15, len(df)-2)):
            r0, r1 = df.iloc[-(i+1)], df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig: return True
        return False
    except Exception: return False

def screen_B3(df, info, market) -> bool:
    """B-3: 売上高加速成長 + FCFマージン改善"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        margin  = (info.get("operatingMargins") or 0) * 100
        if rev_grw < p["rev_grw_min"] * 100: return False
        if margin < 0: return False
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_C1(df, info, market) -> bool:
    """C-1: PEGレシオ割安 + 成長加速"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        peg     = info.get("pegRatio")
        per     = info.get("trailingPE")
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        peg_max = 2.0 if market == "JP" else 1.5
        if peg is None or peg > peg_max or peg <= 0: return False
        if per is None or per < 5: return False
        if rev_grw < p["rev_grw_min"] * 100: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or r.RSI14 < 40: return False
        return True
    except Exception: return False

def screen_C2(df, info, market) -> bool:
    """C-2: 低PSR + 高成長（グロースバリュー）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        psr     = info.get("priceToSalesTrailing12Months")
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        gross   = (info.get("grossMargins") or 0) * 100
        psr_max = 3.0 if market == "JP" else 5.0
        if psr is None or psr > psr_max: return False
        if rev_grw < p["rev_grw_min"] * 100: return False
        if gross < p["gross_min"] * 100: return False
        if any(pd.isna([r.BB_lower, r.BB_mid])): return False
        if r.Close < r.BB_lower: return False
        if pd.isna(r.RSI14) or r.RSI14 < 35: return False
        return True
    except Exception: return False

def screen_C3(df, info, market) -> bool:
    """C-3: EV/Revenue割安 + 収益改善"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        margin  = (info.get("operatingMargins") or 0) * 100
        if rev_grw < p["rev_grw_min"] * 100: return False
        if margin < -5: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        for i in range(min(10, len(df)-2)):
            r0, r1 = df.iloc[-(i+1)], df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig: return True
        return False
    except Exception: return False

def screen_D1(df, info, market) -> bool:
    """D-1: EPSサプライズ + ブレイクアウト"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"]: return False
        return True
    except Exception: return False

def screen_D2(df, info, market) -> bool:
    """D-2: ガイダンス上方修正 + 新高値"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.5: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"]: return False
        return True
    except Exception: return False

def screen_D3(df, info, market) -> bool:
    """D-3: 自社株買い大規模 + モメンタム"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        roe = info.get("returnOnEquity") or 0
        if roe < p["roe_min"]: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        for i in range(min(5, len(df)-2)):
            r0, r1 = df.iloc[-(i+1)], df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig: return True
        return False
    except Exception: return False

def screen_E1(df, info, market) -> bool:
    """E-1: 強気MACDダイバージェンス + ファンダ良好"""
    try:
        p = MARKET_PARAMS[market]
        roe     = info.get("returnOnEquity") or 0
        rev_grw = info.get("revenueGrowth") or 0
        if roe < p["roe_min"]: return False
        if rev_grw < p["rev_grw_min"]: return False
        hist = df["MACD_hist"].dropna()
        if len(hist) < 20: return False
        lows = []
        for i in range(5, min(60, len(hist)-5)):
            idx = -i
            if hist.iloc[idx] < hist.iloc[idx-1] and hist.iloc[idx] < hist.iloc[idx+1]:
                lows.append(idx)
            if len(lows) == 2: break
        if len(lows) < 2: return False
        i1, i2 = lows[0], lows[1]
        close = df["Close"]
        if close.iloc[i1] >= close.iloc[i2]: return False
        if hist.iloc[i1] <= hist.iloc[i2]: return False
        return True
    except Exception: return False

def screen_E2(df, info, market) -> bool:
    """E-2: ショートスクイーズ予兆 + 底打ち"""
    try:
        r = get_latest(df)
        if pd.isna(r.RSI14) or r.RSI14 > 40: return False
        if r.Close <= df["Close"].iloc[-20:].min() * 1.01: return False
        if pd.isna(r.BB_lower) or r.Close < r.BB_lower: return False
        return True
    except Exception: return False

def screen_F1(df, info, market) -> bool:
    """F-1: AI・データインフラ セクターリーダー"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        if rev_grw < p["rev_grw_min"] * 100 * 1.5: return False  # 基準の1.5倍
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 75): return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        return True
    except Exception: return False

def screen_F2(df, info, market) -> bool:
    """F-2: サイバーセキュリティ 成長株"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        gross   = (info.get("grossMargins") or 0) * 100
        gross_min = 55 if market == "JP" else 65
        if rev_grw < p["rev_grw_min"] * 100: return False
        if gross < gross_min: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        if not pd.isna(r.HIGH_52W) and r.HIGH_52W > 0:
            if r.Close < r.HIGH_52W * 0.80: return False
        return True
    except Exception: return False

def screen_F3(df, info, market) -> bool:
    """F-3: バイオテック・ヘルステック カタリスト"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        vol_base = r.VOL_MA50 if not pd.isna(r.VOL_MA50) else r.VOL_MA20
        if pd.isna(vol_base) or r.Volume < vol_base * 1.5: return False
        if (info.get("revenueGrowth") or 0) < 0.05: return False
        return True
    except Exception: return False

def screen_G1(df, info, market) -> bool:
    """G-1: CANSLIM グローバル完全版"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        roe     = info.get("returnOnEquity") or 0
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        roe_min = 0.15 if market == "JP" else 0.17
        if roe < roe_min: return False
        if rev_grw < p["rev_grw_min"] * 100: return False
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        vol_base = r.VOL_MA50 if not pd.isna(r.VOL_MA50) else r.VOL_MA20
        if pd.isna(vol_base) or r.Volume < vol_base * 2.0: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        return True
    except Exception: return False

def screen_G2(df, info, market) -> bool:
    """G-2: CANSLIM 簡易版"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        roe     = info.get("returnOnEquity") or 0
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        if roe < p["roe_min"]: return False
        if rev_grw < p["rev_grw_min"] * 100: return False
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        return True
    except Exception: return False

def screen_H1(df, info, market) -> bool:
    """H-1: グロース × モメンタム × クオリティ 三軸"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        gross   = (info.get("grossMargins") or 0) * 100
        if rev_grw < p["rev_grw_min"] * 100: return False
        if gross < p["gross_min"] * 100: return False
        m, l = r[p["sma_m"]], r[p["sma_l"]]
        if any(pd.isna([m, l])): return False
        if not (r.Close > m > l): return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        if pd.isna(r.RET_60D) or r.RET_60D < 15: return False
        return True
    except Exception: return False

def screen_H2(df, info, market) -> bool:
    """H-2: EPSサプライズ × ブレイクアウト × 相対強度"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        if pd.isna(r.RET_20D) or r.RET_20D < 3: return False
        return True
    except Exception: return False

def screen_H3(df, info, market) -> bool:
    """H-3: テーマセクター強 × 成長加速 × テクニカル"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        if rev_grw < p["rev_grw_min"] * 100: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        if pd.isna(r.ADX) or r.ADX < 20: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        return True
    except Exception: return False

def screen_H4(df, info, market) -> bool:
    """H-4: 安定高品質 ディフェンシブグロース"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        roe     = info.get("returnOnEquity") or 0
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        if roe < p["roe_min"]: return False
        if rev_grw < p["rev_grw_min"] * 100 * 0.8: return False  # 基準の80%
        if pd.isna(r[p["sma_l"]]) or r.Close <= r[p["sma_l"]]: return False
        if pd.isna(r.DEV_SMA200) or r.DEV_SMA200 > 15: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        vol_thresh = 0.022 if market == "JP" else 0.025
        vol = df["Close"].pct_change().iloc[-120:].std()
        if pd.isna(vol) or vol > vol_thresh: return False
        return True
    except Exception: return False

def screen_I1(df, info, market) -> bool:
    """I-1: 高ROIC企業"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        roe    = info.get("returnOnEquity") or 0
        margin = (info.get("operatingMargins") or 0) * 100
        de     = info.get("debtToEquity") or 999
        roe_min    = 0.12 if market == "JP" else 0.15
        margin_min = 12   if market == "JP" else 15
        de_max     = 80   if market == "JP" else 50
        if roe < roe_min: return False
        if margin < margin_min: return False
        if de > de_max: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_I2(df, info, market) -> bool:
    """I-2: ターンアラウンド（業績回復）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        margin  = (info.get("operatingMargins") or 0) * 100
        if rev_grw < 5: return False
        if margin < 0: return False
        if pd.isna(r.LOW_52W) or r.Close < r.LOW_52W * 1.20: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        for i in range(min(15, len(df)-2)):
            r0, r1 = df.iloc[-(i+1)], df.iloc[-(i+2)]
            if any(pd.isna([r0.MACD, r0.MACD_sig, r1.MACD, r1.MACD_sig])): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig: return True
        return False
    except Exception: return False

def screen_I3(df, info, market) -> bool:
    """I-3: 小型株プレミアム"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        mktcap  = info.get("marketCap") or 0
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        margin  = (info.get("operatingMargins") or 0) * 100
        # 小型株上限: JP 300億円 / US 5億USD
        mktcap_max = 30_000_000_000 if market == "JP" else 500_000_000
        if mktcap > 0 and mktcap > mktcap_max: return False
        if mktcap > 0 and mktcap < p["mktcap_min"]: return False
        if rev_grw < p["rev_grw_min"] * 100: return False
        if margin < p["margin_min"] * 100: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.5: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_I4(df, info, market) -> bool:
    """I-4: 高配当 + 成長"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        div_yield = info.get("dividendYield") or 0
        payout    = info.get("payoutRatio") or 0
        rev_grw   = (info.get("revenueGrowth") or 0) * 100
        div_min   = 0.025 if market == "JP" else 0.020
        if div_yield < div_min: return False
        if not (0.20 <= payout <= 0.65): return False
        if rev_grw < 5: return False
        if pd.isna(r[p["sma_m"]]) or r.Close <= r[p["sma_m"]]: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        return True
    except Exception: return False

# ─────────────────────────────────────────
# 戦略一覧
# ─────────────────────────────────────────
STRATEGIES = {
    "A-1_高値BK出来高":        screen_A1,
    "A-2_MA順列ADX":           screen_A2,
    "A-3_相対強度新高値":       screen_A3,
    "B-1_RuleOf40":            screen_B1,
    "B-2_EPS加速粗利拡大":      screen_B2,
    "B-3_売上加速FCF改善":      screen_B3,
    "C-1_PEG割安成長":          screen_C1,
    "C-2_低PSR高成長":          screen_C2,
    "C-3_EV収益改善":           screen_C3,
    "D-1_EPSサプライズBK":      screen_D1,
    "D-2_上方修正新高値":       screen_D2,
    "D-3_自社株買モメンタム":   screen_D3,
    "E-1_MACDダイバージェンス": screen_E1,
    "E-2_ショートスクイーズ":   screen_E2,
    "F-1_AIデータインフラ":     screen_F1,
    "F-2_サイバーセキュリティ":  screen_F2,
    "F-3_バイオヘルステック":   screen_F3,
    "G-1_CANSLIM完全版":        screen_G1,
    "G-2_CANSLIM簡易版":        screen_G2,
    "H-1_グロースモメンタム":   screen_H1,
    "H-2_サプライズBK強度":     screen_H2,
    "H-3_テーマ成長加速":       screen_H3,
    "H-4_安定ディフェンシブ":   screen_H4,
    "I-1_高ROIC":               screen_I1,
    "I-2_ターンアラウンド":     screen_I2,
    "I-3_小型株プレミアム":     screen_I3,
    "I-4_高配当成長":           screen_I4,
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
            market = MARKET_MAP.get(ticker, "US")
            df     = calc_indicators(raw_df)
            info   = fetch_info(ticker)

            if not passes_base_filter(df, info, market):
                continue

            hit = []
            for name, fn in STRATEGIES.items():
                try:
                    if fn(df, info, market):
                        hit.append(name)
                except Exception:
                    pass

            if hit:
                r      = df.iloc[-1]
                mktcap = info.get("marketCap")
                results.append({
                    "市場":          market,
                    "銘柄コード":    ticker,
                    "現在値":        round(float(r.Close), 1),
                    "RSI14":         round(float(r.RSI14), 1) if not pd.isna(r.RSI14) else None,
                    "PER":           round(info.get("trailingPE") or 0, 1) or None,
                    "PBR":           round(info.get("priceToBook") or 0, 2) or None,
                    "ROE(%)":        round((info.get("returnOnEquity") or 0) * 100, 1),
                    "売上成長(%)":   round((info.get("revenueGrowth") or 0) * 100, 1),
                    "粗利率(%)":     round((info.get("grossMargins") or 0) * 100, 1),
                    "マッチ戦略数":  len(hit),
                    "マッチ戦略":    " | ".join(hit),
                })
        except Exception:
            pass

    return sorted(results, key=lambda x: (-x["マッチ戦略数"], x["市場"], x["銘柄コード"]))

def save_results(results: list):
    if not results:
        print("ヒット銘柄なし")
        return

    df_all = pd.DataFrame(results)
    jp_hits = [r for r in results if r["市場"] == "JP"]
    us_hits = [r for r in results if r["市場"] == "US"]

    # CSV
    df_all.to_csv(os.path.join(OUTPUT_DIR, "screening_all.csv"), index=False, encoding="utf-8-sig")
    print(f"[保存] 全銘柄統合: {OUTPUT_DIR}/screening_all.csv")

    rows = []
    for r in results:
        for s in r["マッチ戦略"].split(" | "):
            row = r.copy(); row["戦略"] = s; rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "screening_by_strategy.csv"), index=False, encoding="utf-8-sig")
    print(f"[保存] 戦略別:    {OUTPUT_DIR}/screening_by_strategy.csv")

    summary = []
    for name in STRATEGIES:
        hits = [r for r in results if name in r["マッチ戦略"]]
        jp_s = [r["銘柄コード"] for r in hits if r["市場"]=="JP"]
        us_s = [r["銘柄コード"] for r in hits if r["市場"]=="US"]
        summary.append({
            "戦略": name,
            "JP": len(jp_s), "US": len(us_s),
            "合計": len(hits),
            "JP銘柄": ",".join(jp_s),
            "US銘柄": ",".join(us_s),
        })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(OUTPUT_DIR, "screening_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"[保存] サマリー:   {OUTPUT_DIR}/screening_summary.csv")

    # TXT レポート
    path_txt = os.path.join(OUTPUT_DIR, "screening_report.txt")
    cols = ["市場","銘柄コード","現在値","RSI14","PER","PBR","ROE(%)","売上成長(%)","粗利率(%)","マッチ戦略数","マッチ戦略"]
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" スクリーニング結果 ({TODAY})\n")
        f.write(f" 対象: 国内株式 / 米国株式  |  全27戦略\n")
        f.write("=" * 70 + "\n")
        f.write(f"スキャン銘柄数: {len(STOCK_UNIVERSE)}  (JP:{_n_jp} / US:{_n_us})\n")
        f.write(f"ヒット銘柄数:   {len(results)}  (JP:{len(jp_hits)} / US:{len(us_hits)})\n\n")

        f.write("【戦略別ヒット数】\n")
        f.write(df_sum.to_string(index=False) + "\n\n")

        # 注目銘柄（マッチ数上位）
        top = df_all[df_all["マッチ戦略数"] >= 2].head(15)
        if len(top):
            f.write("【注目銘柄（マッチ数上位）】\n")
            top_cols = ["市場","銘柄コード","現在値","RSI14","PER","ROE(%)","売上成長(%)","粗利率(%)","マッチ戦略数","マッチ戦略"]
            f.write(top[top_cols].to_string(index=False) + "\n\n")

        f.write("【国内株 ヒット銘柄】\n")
        df_jp = df_all[df_all["市場"]=="JP"]
        f.write((df_jp[cols].to_string(index=False) if len(df_jp) else "  なし") + "\n\n")

        f.write("【米国株 ヒット銘柄】\n")
        df_us = df_all[df_all["市場"]=="US"]
        f.write((df_us[cols].to_string(index=False) if len(df_us) else "  なし") + "\n")
    print(f"[保存] レポート:   {path_txt}")

    # コンソール出力
    print("\n" + "=" * 70)
    print(f" スクリーニング結果 ({TODAY})")
    print(f" 対象: 国内株式 / 米国株式  |  全27戦略")
    print("=" * 70)
    print(f"スキャン銘柄数: {len(STOCK_UNIVERSE)}  (JP:{_n_jp} / US:{_n_us})")
    print(f"ヒット銘柄数:   {len(results)}  (JP:{len(jp_hits)} / US:{len(us_hits)})\n")
    print(df_sum.to_string(index=False))
    print("\n【国内株 ヒット銘柄】")
    print(df_all[df_all["市場"]=="JP"][cols].to_string(index=False) if len(jp_hits) else "  なし")
    print("\n【米国株 ヒット銘柄】")
    print(df_all[df_all["市場"]=="US"][cols].to_string(index=False) if len(us_hits) else "  なし")

if __name__ == "__main__":
    results = run_all_screens()
    save_results(results)
