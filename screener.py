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

# ── PDF 生成 ──────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
                                 Spacer, PageBreak, HRFlowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 日本語フォント設定（Yu Gothic / BIZ-UDGothic / CIDフォント の順で試行）
_FONT_CANDIDATES = [
    ("/mnt/c/Windows/Fonts/YuGothR.ttc",  "YuGothic",     "YuGothic"),
    ("/mnt/c/Windows/Fonts/YuGothB.ttc",  "YuGothicBold", "YuGothicBold"),
    ("/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc", "BIZGothic",     "BIZGothic"),
    ("/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc", "BIZGothicBold", "BIZGothicBold"),
]
_FONT_NORMAL = "HeiseiKakuGo-W5"
_FONT_BOLD   = "HeiseiKakuGo-W5"
_using_ttf   = False

_reg_normal, _reg_bold = None, None
for _path, _reg, _ in _FONT_CANDIDATES:
    if os.path.exists(_path):
        try:
            pdfmetrics.registerFont(TTFont(_reg, _path))
            if _reg_normal is None:
                _reg_normal = _reg
            elif _reg_bold is None:
                _reg_bold = _reg
        except Exception:
            pass

if _reg_normal:
    _FONT_NORMAL = _reg_normal
    _FONT_BOLD   = _reg_bold or _reg_normal
    _using_ttf   = True
    # フォントファミリー登録（ps2tt対応）
    pdfmetrics.registerFontFamily(_reg_normal,
                                  normal=_reg_normal,
                                  bold=_FONT_BOLD,
                                  italic=_reg_normal,
                                  boldItalic=_FONT_BOLD)
else:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))

# カラーパレット
C_NAVY   = colors.HexColor("#1a3a5c")
C_BLUE   = colors.HexColor("#2563a8")
C_LBLUE  = colors.HexColor("#dbeafe")
C_ORANGE = colors.HexColor("#f57c00")
C_GREEN  = colors.HexColor("#15803d")
C_RED    = colors.HexColor("#dc2626")
C_LGRAY  = colors.HexColor("#f5f5f5")
C_WHITE  = colors.white
C_BLACK  = colors.black

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
# グロース重視スコアリング & リスク判定
# ─────────────────────────────────────────
# スコアを2倍にするグロース系戦略
GROWTH_STRATEGIES = {
    "A-1","A-2","A-3",          # モメンタム
    "B-1","B-2","B-3",          # グロース×クオリティ
    "F-1","F-2",                 # テーマ成長
    "G-1","G-2",                 # CANSLIM
    "H-1","H-3",                 # 複合グロース
    "I-1","I-3",                 # 高ROIC・小型プレミアム
}

def calc_growth_score(r: dict) -> float:
    """グロース重視スコア（高いほど推奨度高）"""
    score = 0.0
    for s in r["マッチ戦略"].split(" | "):
        score += 2.0 if s[:3] in GROWTH_STRATEGIES else 1.0

    rev_grw = r.get("売上成長(%)", 0) or 0
    if   rev_grw >= 50: score += 5
    elif rev_grw >= 30: score += 3
    elif rev_grw >= 20: score += 2
    elif rev_grw >= 10: score += 1

    gross = r.get("粗利率(%)", 0) or 0
    if   gross >= 70: score += 3
    elif gross >= 50: score += 2
    elif gross >= 35: score += 1

    roe = r.get("ROE(%)", 0) or 0
    if   roe >= 30: score += 2
    elif roe >= 15: score += 1

    rsi = r.get("RSI14") or 0
    if 52 <= rsi <= 72: score += 1   # 上昇トレンド理想ゾーン

    return round(score, 1)


def is_high_risk(df: pd.DataFrame, info: dict) -> bool:
    """ハイリスク銘柄判定（True = 除外対象）"""
    try:
        latest = df.iloc[-1]
        rsi = float(latest.RSI14) if not pd.isna(latest.RSI14) else 0
        if rsi > 80: return True                          # 過熱

        if not pd.isna(latest.DEV_SMA200) and latest.DEV_SMA200 > 60:
            return True                                   # 200日線から60%超乖離

        vol = df["Close"].pct_change().iloc[-60:].std() * np.sqrt(252)
        if not np.isnan(vol) and vol > 0.85: return True  # 年率85%超のボラ
    except Exception:
        pass
    return False


def generate_buy_reasons(r: dict, df: pd.DataFrame, info: dict, market: str):
    """(推奨理由リスト, リスク要因リスト, シナリオ文字列) を返す"""
    reasons, risks = [], []

    rev_grw = r.get("売上成長(%)", 0) or 0
    gross   = r.get("粗利率(%)", 0) or 0
    roe     = r.get("ROE(%)", 0) or 0
    rsi     = r.get("RSI14") or 0
    per     = r.get("PER") or 0
    matched = r["マッチ戦略"].split(" | ")

    # ── グロース理由 ──
    if rev_grw >= 30:
        reasons.append(f"売上高 {rev_grw:.0f}%成長（高成長フェーズ継続中）")
    elif rev_grw >= 15:
        reasons.append(f"売上高 {rev_grw:.0f}%成長（安定高成長）")

    if gross >= 60:
        reasons.append(f"粗利率 {gross:.0f}%（高収益ビジネスモデル）")
    elif gross >= 40:
        reasons.append(f"粗利率 {gross:.0f}%（スケーラブルな事業構造）")

    if roe >= 20:
        reasons.append(f"ROE {roe:.0f}%（高い資本効率・株主価値向上力）")
    elif roe >= 12:
        reasons.append(f"ROE {roe:.0f}%（安定した収益力）")

    # ── テクニカル理由 ──
    if 55 <= rsi <= 72:
        reasons.append(f"RSI {rsi:.0f}（上昇トレンドの理想ゾーン・過熱感なし）")
    try:
        latest = df.iloc[-1]
        p = MARKET_PARAMS[market]
        sm, sl = float(latest[p["sma_m"]]), float(latest[p["sma_l"]])
        cl = float(latest.Close)
        if cl > sm > sl:
            reasons.append("株価が中期・長期移動平均線の上位に位置（上昇トレンド確認）")
        adx = float(latest.ADX) if not pd.isna(latest.ADX) else 0
        if adx > 28:
            reasons.append(f"ADX {adx:.0f}（強いトレンドの存在を示す）")
        ret60 = float(latest.RET_60D) if not pd.isna(latest.RET_60D) else 0
        if ret60 > 15:
            reasons.append(f"直近60日で {ret60:.0f}%上昇（高い相対強度）")
    except Exception:
        pass

    # ── 戦略シグナル理由 ──
    _key = {
        "B-1": "Rule of 40達成（成長率＋利益率≧40%）",
        "G-1": "CANSLIM全条件クリア（最強グロースシグナル）",
        "H-1": "グロース×モメンタム×クオリティ三軸クリア",
        "A-1": "52週高値ブレイクアウト＋出来高急増",
        "F-1": "AI・データインフラ テーマリーダー",
        "I-1": "高ROIC（資本効率最高クラス）",
        "A-2": "多重MA完全順列＋強トレンド確認",
    }
    for s in matched:
        if s[:3] in _key:
            reasons.append(_key[s[:3]])
            break

    # ── リスク要因 ──
    if rsi > 70:
        risks.append(f"RSI {rsi:.0f}（やや過熱気味・短期調整に注意）")
    if per and per > 80:
        risks.append(f"PER {per:.0f}倍（高バリュエーション・業績の高期待が織り込まれている）")
    try:
        vol = df["Close"].pct_change().iloc[-60:].std() * np.sqrt(252) * 100
        if vol > 50:
            risks.append(f"年率ボラティリティ {vol:.0f}%（価格変動が大きい）")
    except Exception:
        pass
    if not risks:
        risks.append("特段の高リスク要因なし（ベースフィルター・ハイリスク除外済）")

    # ── シナリオ ──
    try:
        cl = float(df.iloc[-1].Close)
        h52 = float(df["Close"].rolling(252).max().iloc[-1])
        tgt = max(h52 * 1.05, cl * 1.20)
        stop = cl * 0.88
        currency = "円" if market == "JP" else "USD"
        scenario = (f"目標: {tgt:.1f}{currency} ({(tgt/cl-1)*100:+.0f}%) /"
                    f" 撤退ライン: {stop:.1f}{currency} (-12%)")
    except Exception:
        scenario = "目標・撤退ラインはご自身でご判断ください"

    return reasons[:5], risks[:3], scenario


# ─────────────────────────────────────────
# PDF ヘルパー
# ─────────────────────────────────────────

# セクター・業種 日本語マッピング
_SECTOR_JP = {
    "Technology": "テクノロジー",
    "Healthcare": "ヘルスケア",
    "Financial Services": "金融サービス",
    "Consumer Cyclical": "消費財（景気敏感）",
    "Consumer Defensive": "消費財（ディフェンシブ）",
    "Industrials": "産業・製造",
    "Basic Materials": "素材・化学",
    "Energy": "エネルギー",
    "Real Estate": "不動産",
    "Communication Services": "通信・メディア",
    "Utilities": "公益事業",
    "Software—Application": "ソフトウェア（アプリ）",
    "Software - Application": "ソフトウェア（アプリ）",
    "Semiconductors": "半導体",
    "Semiconductor Equipment & Materials": "半導体製造装置",
    "Drug Manufacturers - General": "製薬（大手）",
    "Biotechnology": "バイオテクノロジー",
    "Internet Content & Information": "インターネット・情報",
    "Electronic Components": "電子部品",
    "Insurance - Property & Casualty": "損害保険",
    "Specialty Retail": "専門小売",
    "Restaurants": "飲食チェーン",
    "Medical Devices": "医療機器",
    "Asset Management": "資産運用",
    "Banks—Regional": "地方銀行",
    "Banks—Diversified": "総合銀行",
    "Telecom Services": "通信サービス",
    "Auto Manufacturers": "自動車メーカー",
    "Machinery": "機械",
    "Wholesale—Specialty": "専門商社",
    "Electronics": "エレクトロニクス",
    "Information Technology Services": "ITサービス",
    "Computer Hardware": "コンピュータハードウェア",
}

def _jp_sector(name: str) -> str:
    """セクター/業種名を日本語に変換（マッピングになければ原文）"""
    return _SECTOR_JP.get(name or "", name or "N/A")


def _safe(text: str) -> str:
    """reportlab Paragraph用にHTML特殊文字をエスケープ"""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# テーブルセル用 ParagraphStyle（モジュールレベルで定義）
_cs_hdr   = ParagraphStyle("_csh",  fontName=_FONT_BOLD,   fontSize=8.5,
                             leading=12, textColor=C_WHITE,  wordWrap="CJK")
_cs_body  = ParagraphStyle("_csb",  fontName=_FONT_NORMAL, fontSize=8.5,
                             leading=12, textColor=C_BLACK,  wordWrap="CJK")
_cs_lblue = ParagraphStyle("_cslb", fontName=_FONT_BOLD,   fontSize=8.5,
                             leading=12, textColor=C_NAVY,   wordWrap="CJK")


def _style(name, **kw):
    base = dict(fontName=_FONT_NORMAL, fontSize=10, leading=16,
                textColor=C_BLACK, spaceAfter=4, wordWrap="CJK")
    base.update(kw)
    return ParagraphStyle(name, **base)


def _p(text, s):
    return Paragraph(_safe(text), s)


def _tbl(data, col_widths, extra_styles=None, hdr_bg=C_NAVY, subhdr_rows=None):
    """
    テキスト折り返し対応テーブル。
    セルは Paragraph に変換して wordWrap='CJK' を保証する。
    hdr_bg  : ヘッダー行の背景色
    subhdr_rows : (row_index, bg_color) のリスト（サブヘッダー行）
    """
    processed = []
    for ri, row in enumerate(data):
        new_row = []
        for cell in row:
            if isinstance(cell, Paragraph):
                new_row.append(cell)
            else:
                st = _cs_hdr if ri == 0 else _cs_body
                new_row.append(Paragraph(_safe(cell), st))
        processed.append(new_row)

    t = Table(processed, colWidths=col_widths, repeatRows=1)
    base = [
        ("BACKGROUND",    (0, 0), (-1, 0), hdr_bg),
        ("ROWBACKGROUNDS",(0, 1), (-1,-1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1,-1), 4),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 4),
        ("LEFTPADDING",   (0, 0), (-1,-1), 5),
        ("RIGHTPADDING",  (0, 0), (-1,-1), 5),
    ]
    if subhdr_rows:
        for row_i, bg in subhdr_rows:
            base += [
                ("BACKGROUND", (0, row_i), (-1, row_i), bg),
            ]
            # サブヘッダー行のセルを lblue スタイルに上書き
            for ci in range(len(processed[row_i])):
                if not isinstance(data[row_i][ci], Paragraph):
                    processed[row_i][ci] = Paragraph(
                        _safe(str(data[row_i][ci])), _cs_lblue)
    if extra_styles:
        base += extra_styles
    t.setStyle(TableStyle(base))
    return t


# ─────────────────────────────────────────
# メイン実行
# ─────────────────────────────────────────
def run_all_screens():
    """スクリーニング実行。(results, stock_data_raw) を返す"""
    print("\nスクリーニング実行中...")
    stock_data_raw = fetch_stock_data(STOCK_UNIVERSE, period="2y")

    results = []
    for ticker, raw_df in stock_data_raw.items():
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
                r   = df.iloc[-1]
                rec = {
                    "市場":          market,
                    "銘柄コード":    ticker,
                    "銘柄名":        info.get("longName") or info.get("shortName") or ticker,
                    "セクター":      info.get("sector") or "N/A",
                    "現在値":        round(float(r.Close), 1),
                    "RSI14":         round(float(r.RSI14), 1) if not pd.isna(r.RSI14) else None,
                    "PER":           round(info.get("trailingPE") or 0, 1) or None,
                    "PBR":           round(info.get("priceToBook") or 0, 2) or None,
                    "ROE(%)":        round((info.get("returnOnEquity") or 0) * 100, 1),
                    "売上成長(%)":   round((info.get("revenueGrowth") or 0) * 100, 1),
                    "粗利率(%)":     round((info.get("grossMargins") or 0) * 100, 1),
                    "営業利益率(%)": round((info.get("operatingMargins") or 0) * 100, 1),
                    "マッチ戦略数":  len(hit),
                    "マッチ戦略":    " | ".join(hit),
                    "ハイリスク":    is_high_risk(df, info),
                    "_info":         info,
                }
                rec["グロース評価スコア"] = calc_growth_score(rec)
                results.append(rec)
        except Exception:
            pass

    results_sorted = sorted(results,
                            key=lambda x: (-x["グロース評価スコア"], x["市場"], x["銘柄コード"]))
    return results_sorted, stock_data_raw


# ─────────────────────────────────────────
# PDF出力①: 分析レポート
# ─────────────────────────────────────────
def save_analysis_pdf(results: list, stock_data: dict):
    """スクリーニング全体の分析レポートPDF"""
    if not results:
        print("ヒット銘柄なし"); return

    path = os.path.join(OUTPUT_DIR, "analysis_report.pdf")
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)

    W = 180  # mm（有効幅）

    # ── スタイル ──
    s_title = _style("a_title", fontSize=20, fontName=_FONT_BOLD,
                     textColor=C_NAVY, spaceAfter=6, leading=28)
    s_sub   = _style("a_sub",   fontSize=12, fontName=_FONT_BOLD,
                     textColor=C_BLUE, spaceAfter=4, leading=18)
    s_body  = _style("a_body",  fontSize=9,  leading=14, spaceAfter=3)
    s_small = _style("a_sm",    fontSize=8,  leading=12, spaceAfter=2,
                     textColor=colors.HexColor("#555555"))
    s_en    = _style("a_en",    fontSize=7.5, leading=11, spaceAfter=2,
                     textColor=colors.HexColor("#666666"))  # 英語原文用

    jp_hits = [r for r in results if r["市場"] == "JP"]
    us_hits = [r for r in results if r["市場"] == "US"]
    top     = [r for r in results if r["グロース評価スコア"] >= 5]

    elems = []

    # ━━ 表紙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elems += [
        Spacer(1, 18*mm),
        _p("株式スクリーニング　分析レポート", s_title),
        _p(f"実行日: {TODAY}　　対象: 国内株式 / 米国株式　　全27戦略", s_body),
        HRFlowable(width="100%", thickness=2, color=C_NAVY, spaceAfter=8),
        Spacer(1, 5*mm),
    ]

    sum_data = [
        ["項目", "件数"],
        ["スキャン銘柄数（国内株）",   f"{_n_jp}銘柄"],
        ["スキャン銘柄数（米国株）",   f"{_n_us}銘柄"],
        ["ヒット銘柄数（国内株）",     f"{len(jp_hits)}銘柄"],
        ["ヒット銘柄数（米国株）",     f"{len(us_hits)}銘柄"],
        ["注目銘柄（グロース評価スコア5以上）", f"{len(top)}銘柄"],
    ]
    elems.append(_tbl(sum_data, [130*mm, 50*mm]))
    elems.append(Spacer(1, 8*mm))

    # ━━ 戦略別ヒット数 ━━━━━━━━━━━━━━━━━━━━━
    elems += [
        _p("■ 戦略別ヒット数", s_sub),
        HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=4),
    ]
    st_data = [["戦略コード・名称", "国内", "米国", "合計", "国内ヒット銘柄", "米国ヒット銘柄"]]
    for sname in STRATEGIES:
        hits = [r for r in results if sname in r["マッチ戦略"]]
        jp_s = [r["銘柄コード"] for r in hits if r["市場"] == "JP"]
        us_s = [r["銘柄コード"] for r in hits if r["市場"] == "US"]
        st_data.append([sname, str(len(jp_s)), str(len(us_s)), str(len(hits)),
                        "、".join(jp_s) or "－", "、".join(us_s) or "－"])
    # 列幅: 56+10+10+10+47+47=180mm
    elems.append(_tbl(st_data, [56*mm, 10*mm, 10*mm, 10*mm, 47*mm, 47*mm]))
    elems.append(PageBreak())

    # ━━ ヒット銘柄一覧 ━━━━━━━━━━━━━━━━━━━━━
    elems += [
        _p("■ ヒット銘柄一覧（グロース評価スコア順）", s_sub),
        HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=4),
        _p("※ グロース評価スコア: 成長系戦略2倍加重 + 売上成長・粗利率・ROEボーナス", s_small),
        _p("※ リスク欄: ○=通常　⚠=高リスク（RSI過熱・大幅乖離・高ボラ）", s_small),
        Spacer(1, 2*mm),
    ]
    all_data = [["市場", "コード", "銘柄名", "スコア", "現在値",
                 "RSI", "売上成長%", "粗利率%", "ROE%", "マッチ数", "リスク"]]
    for r in results:
        name_s   = (r.get("銘柄名") or r["銘柄コード"])[:16]
        risk_str = "⚠高" if r.get("ハイリスク") else "○"
        all_data.append([
            r["市場"], r["銘柄コード"], name_s,
            str(r["グロース評価スコア"]),
            str(r["現在値"]),
            str(r.get("RSI14") or "－"),
            f"{r.get('売上成長(%)',0):.0f}%",
            f"{r.get('粗利率(%)',0):.0f}%",
            f"{r.get('ROE(%)',0):.0f}%",
            str(r["マッチ戦略数"]),
            risk_str,
        ])
    # 列幅: 10+18+40+13+20+10+17+14+12+14+12=180mm
    elems.append(_tbl(all_data,
                      [10*mm, 18*mm, 40*mm, 13*mm, 20*mm,
                       10*mm, 17*mm, 14*mm, 12*mm, 14*mm, 12*mm]))
    elems.append(PageBreak())

    # ━━ 注目銘柄 詳細分析 ━━━━━━━━━━━━━━━━━━
    elems += [
        _p("■ 注目銘柄　詳細分析", s_sub),
        HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=6),
    ]

    detail_targets = [r for r in results
                      if r["グロース評価スコア"] >= 4 and not r.get("ハイリスク")][:12]
    print(f"\n詳細分析生成中... ({len(detail_targets)}銘柄)")

    for i, r in enumerate(detail_targets, 1):
        ticker  = r["銘柄コード"]
        market  = r["市場"]
        info    = r.get("_info") or fetch_info(ticker)
        print(f"  [{i}/{len(detail_targets)}] {ticker} ...")

        name     = info.get("longName") or info.get("shortName") or ticker
        sector   = _jp_sector(info.get("sector"))
        industry = _jp_sector(info.get("industry"))
        mktcap   = info.get("marketCap") or 0
        mktcap_s = (f"{mktcap/1e8:.0f}億円" if market=="JP" else f"${mktcap/1e9:.1f}B") if mktcap else "N/A"
        per_v    = info.get("trailingPE")
        pbr_v    = info.get("priceToBook")
        roe_v    = (info.get("returnOnEquity")   or 0) * 100
        roa_v    = (info.get("returnOnAssets")   or 0) * 100
        gross_v  = (info.get("grossMargins")     or 0) * 100
        op_v     = (info.get("operatingMargins") or 0) * 100
        rev_v    = (info.get("revenueGrowth")    or 0) * 100
        div_v    = (info.get("dividendYield")    or 0) * 100
        de_v     = info.get("debtToEquity")
        desc_en  = (info.get("longBusinessSummary") or "")[:280]

        currency = "円" if market == "JP" else "USD"

        # 銘柄ヘッダー（ユニーク名を使用）
        hd_style = _style(f"a_hd{i}", fontSize=11, fontName=_FONT_BOLD,
                          textColor=C_NAVY, spaceAfter=2, leading=16)
        sm_style = _style(f"a_sm{i}", fontSize=8, leading=12, spaceAfter=2,
                          textColor=colors.HexColor("#555555"))
        elems += [
            _p(f"【{i}】{ticker}　{name}", hd_style),
            _p(f"市場: {'国内株（東証）' if market=='JP' else '米国株（NYSE/NASDAQ）'}　"
               f"セクター: {sector}　業種: {industry}　"
               f"グロース評価スコア: {r['グロース評価スコア']}", sm_style),
            Spacer(1, 1*mm),
        ]

        # 事業概要（英語原文）
        if desc_en:
            elems += [
                _p("◆ 事業概要（英語）", _style(f"a_dh{i}", fontName=_FONT_BOLD,
                   fontSize=8, leading=12, textColor=C_BLUE, spaceAfter=1)),
                _p(desc_en + ("…" if len(info.get("longBusinessSummary","")) > 280 else ""),
                   s_en),
                Spacer(1, 1*mm),
            ]

        # 財務指標テーブル（2列×6行）
        fin_data = [
            ["財務指標", "数値", "財務指標", "数値"],
            ["時価総額",   mktcap_s,
             "売上成長率（前年比）", f"{rev_v:.1f}%"],
            ["株価収益率（PER）", f"{per_v:.1f}倍" if per_v else "N/A",
             "粗利率",     f"{gross_v:.1f}%"],
            ["株価純資産倍率（PBR）", f"{pbr_v:.2f}倍" if pbr_v else "N/A",
             "営業利益率", f"{op_v:.1f}%"],
            ["自己資本利益率（ROE）", f"{roe_v:.1f}%",
             "総資産利益率（ROA）",  f"{roa_v:.1f}%"],
            ["配当利回り", f"{div_v:.2f}%",
             "負債資本比率（D/E）",  f"{de_v:.1f}" if de_v else "N/A"],
        ]
        # 列幅: 50+35+50+45=180mm
        elems.append(_tbl(fin_data, [50*mm, 35*mm, 50*mm, 45*mm]))
        elems.append(Spacer(1, 2*mm))

        # 株価動向テーブル（5年）
        try:
            df5 = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if df5 is not None and len(df5) >= 30:
                df5.columns = [c[0] if isinstance(c, tuple) else c for c in df5.columns]
                cl5 = df5["Close"].squeeze()
                cur = float(cl5.iloc[-1])
                h52 = float(cl5.rolling(252).max().iloc[-1])
                l52 = float(cl5.rolling(252).min().iloc[-1])

                ret_lines = []
                for y in [1, 2, 3]:
                    lb = y * 252
                    if len(cl5) > lb:
                        ret = (cur / float(cl5.iloc[-lb]) - 1) * 100
                        ret_lines.append(
                            f"{y}年前比: {'▲' if ret>0 else '▼'}{abs(ret):.0f}%")

                price_data = [
                    ["株価動向（過去5年）", ""],
                    ["52週高値",
                     f"{h52:.1f}{currency}　（現在比 {(cur/h52-1)*100:+.0f}%）"],
                    ["52週安値",
                     f"{l52:.1f}{currency}　（現在比 {(cur/l52-1)*100:+.0f}%）"],
                    ["年次リターン",
                     "　　".join(ret_lines) or "データ不足"],
                ]
                # 行0をサブヘッダーとして処理
                elems.append(_tbl(price_data, [50*mm, 130*mm],
                                  extra_styles=[
                                      ("SPAN",       (0,0), (-1,0)),
                                      ("BACKGROUND", (0,0), (-1,0), C_LBLUE),
                                  ],
                                  subhdr_rows=[(0, C_LBLUE)]))
                elems.append(Spacer(1, 2*mm))
        except Exception:
            pass

        # 直近ニュース
        try:
            news_list   = yf.Ticker(ticker).news or []
            news_shown  = []
            for n in news_list[:15]:
                title  = n.get("title") or (n.get("content") or {}).get("title", "")
                pub_ts = n.get("providerPublishTime") or 0
                pub_dt = (datetime.fromtimestamp(pub_ts).strftime("%Y/%m/%d")
                          if pub_ts else "－")
                if title:
                    news_shown.append([pub_dt, title[:72]])
                if len(news_shown) >= 5: break
            if news_shown:
                nd = [["直近ニュース（英語）"]] + \
                     [[f"[{d}]　{t}"] for d, t in news_shown]
                elems.append(_tbl(nd, [180*mm],
                                  subhdr_rows=[(0, C_LBLUE)],
                                  extra_styles=[("SPAN",(0,0),(-1,0))]))
        except Exception:
            pass

        elems += [
            Spacer(1, 5*mm),
            HRFlowable(width="100%", thickness=0.3,
                       color=colors.HexColor("#cccccc"), spaceAfter=5),
        ]
        time.sleep(0.2)

    doc.build(elems)
    print(f"[保存] 分析レポート: {path}")


# ─────────────────────────────────────────
# PDF出力②: 今買うべき銘柄レポート
# ─────────────────────────────────────────
def save_recommendation_pdf(results: list, stock_data: dict):
    """今買うべき銘柄レポートPDF（グロース重視・ハイリスク除外）"""
    candidates = [r for r in results if not r.get("ハイリスク") and r["マッチ戦略数"] >= 2]
    candidates = sorted(candidates, key=lambda x: -x["グロース評価スコア"])[:10]

    if not candidates:
        print("推奨候補なし"); return

    path = os.path.join(OUTPUT_DIR, "buy_recommendation.pdf")
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)

    # ── スタイル ──
    s_title    = _style("r_title", fontSize=22, fontName=_FONT_BOLD,
                        textColor=C_NAVY, spaceAfter=6, leading=30)
    s_sub      = _style("r_sub",   fontSize=12, fontName=_FONT_BOLD,
                        textColor=C_BLUE, spaceAfter=4, leading=18)
    s_body     = _style("r_body",  fontSize=9,  leading=14, spaceAfter=3)
    s_small    = _style("r_sm",    fontSize=8,  leading=12, spaceAfter=2,
                        textColor=colors.HexColor("#555555"))
    s_green    = _style("r_grn",   fontSize=10, leading=15, leftIndent=10,
                        spaceAfter=3, textColor=C_GREEN)
    s_risk     = _style("r_rsk",   fontSize=9,  leading=14, leftIndent=10,
                        spaceAfter=2, textColor=C_RED)
    s_scenario = _style("r_sc",    fontSize=10, fontName=_FONT_BOLD, leading=16,
                        textColor=C_ORANGE, spaceAfter=4)
    s_match    = _style("r_mt",    fontSize=8.5, leading=13, spaceAfter=3,
                        textColor=colors.HexColor("#333333"))

    elems = []

    # ━━ 表紙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elems += [
        Spacer(1, 14*mm),
        _p("今買うべき銘柄レポート", s_title),
        _p("グロース重視　×　ハイリスク除外　×　中期投資", s_sub),
        _p(f"実行日: {TODAY}", s_body),
        HRFlowable(width="100%", thickness=2, color=C_ORANGE, spaceAfter=8),
        Spacer(1, 4*mm),
        _p("【選定基準】", _style("r_sc0", fontName=_FONT_BOLD, fontSize=10,
           textColor=C_NAVY, spaceAfter=3, leading=15)),
        _p("・ 27戦略スクリーニングで2戦略以上にヒット", s_body),
        _p("・ グロース評価スコア上位（売上成長・粗利率・ROE・モメンタムを総合評価）", s_body),
        _p("・ ハイリスク銘柄を除外"
           "（RSI過熱 / 200日移動平均線から60%超乖離 / 年率ボラティリティ85%超）", s_body),
        _p("・ 推奨順位はグロース評価スコア降順", s_body),
        Spacer(1, 6*mm),
        _p("【推奨銘柄一覧】", _style("r_sc1", fontName=_FONT_BOLD, fontSize=10,
           textColor=C_NAVY, spaceAfter=3, leading=15)),
    ]

    # 推奨銘柄概要テーブル
    ov_data = [["順位", "市場", "コード", "銘柄名",
                "スコア", "現在値", "売上成長%", "粗利率%", "ROE%"]]
    for rank, r in enumerate(candidates, 1):
        name_s   = (r.get("銘柄名") or r["銘柄コード"])[:18]
        currency = "円" if r["市場"] == "JP" else "$"
        ov_data.append([
            f"#{rank}", r["市場"], r["銘柄コード"], name_s,
            str(r["グロース評価スコア"]),
            f"{currency}{r['現在値']}",
            f"{r.get('売上成長(%)',0):.0f}%",
            f"{r.get('粗利率(%)',0):.0f}%",
            f"{r.get('ROE(%)',0):.0f}%",
        ])
    # 列幅: 10+12+18+44+14+22+20+18+22=180mm
    elems.append(_tbl(ov_data, [10*mm, 12*mm, 18*mm, 44*mm, 14*mm,
                                 22*mm, 20*mm, 18*mm, 22*mm]))
    elems.append(PageBreak())

    # ━━ 各銘柄詳細 ━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n推奨レポート生成中... ({len(candidates)}銘柄)")

    for rank, r in enumerate(candidates, 1):
        ticker  = r["銘柄コード"]
        market  = r["市場"]
        info    = r.get("_info") or fetch_info(ticker)
        print(f"  [{rank}/{len(candidates)}] {ticker} ...")

        name     = info.get("longName") or info.get("shortName") or ticker
        sector   = _jp_sector(info.get("sector"))
        currency = "円" if market == "JP" else "USD"
        mktcap   = info.get("marketCap") or 0
        mktcap_s = (f"{mktcap/1e8:.0f}億円" if market=="JP" else f"${mktcap/1e9:.1f}B") if mktcap else "N/A"
        per_v    = info.get("trailingPE")
        pbr_v    = info.get("priceToBook")
        roe_v    = (info.get("returnOnEquity")   or 0) * 100
        rev_v    = (info.get("revenueGrowth")    or 0) * 100
        gross_v  = (info.get("grossMargins")     or 0) * 100
        op_v     = (info.get("operatingMargins") or 0) * 100

        # 推奨理由・リスク・シナリオ
        raw_df = stock_data.get(ticker)
        if raw_df is not None:
            df_ind = calc_indicators(raw_df)
        else:
            try:
                raw_df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
                raw_df.columns = [c[0] if isinstance(c, tuple) else c for c in raw_df.columns]
                df_ind = calc_indicators(raw_df)
            except Exception:
                df_ind = pd.DataFrame()

        reasons, risks, scenario = generate_buy_reasons(r, df_ind, info, market)

        # ── 銘柄ヘッダー ──
        hd_s = _style(f"r_hd{rank}", fontSize=13, fontName=_FONT_BOLD,
                      textColor=C_NAVY, spaceAfter=2, leading=19)
        sm_s = _style(f"r_sm{rank}", fontSize=8.5, leading=13, spaceAfter=1,
                      textColor=colors.HexColor("#444444"))
        elems += [
            _p(f"推奨 #{rank}　{ticker}　{name}", hd_s),
            _p(f"市場: {'国内株（東証）' if market=='JP' else '米国株（NYSE/NASDAQ）'}　"
               f"セクター: {sector}　"
               f"現在値: {r['現在値']}{currency}　"
               f"グロース評価スコア: {r['グロース評価スコア']}", sm_s),
            HRFlowable(width="100%", thickness=1.2, color=C_ORANGE, spaceAfter=5),
        ]

        # ── 今買うべき理由 ──
        elems.append(_p("【今買うべき理由】",
                        _style(f"r_rh{rank}", fontName=_FONT_BOLD, fontSize=11,
                               textColor=C_GREEN, spaceAfter=3, leading=16)))
        for reason in reasons:
            elems.append(_p(f"✓　{reason}", s_green))

        elems.append(Spacer(1, 3*mm))

        # ── マッチ戦略シグナル ──
        elems += [
            _p("【マッチした戦略シグナル】",
               _style(f"r_mh{rank}", fontName=_FONT_BOLD, fontSize=10,
                      textColor=C_BLUE, spaceAfter=2, leading=14)),
            _p(r["マッチ戦略"].replace(" | ", "　/　"), s_match),
            Spacer(1, 3*mm),
        ]

        # ── 財務指標テーブル ──
        fin_data = [
            ["時価総額", "PER（株価収益率）", "PBR（株価純資産倍率）",
             "ROE（自己資本利益率）", "売上成長率", "粗利率", "営業利益率"],
            [mktcap_s,
             f"{per_v:.0f}倍" if per_v else "N/A",
             f"{pbr_v:.1f}倍" if pbr_v else "N/A",
             f"{roe_v:.0f}%",
             f"{rev_v:.0f}%",
             f"{gross_v:.0f}%",
             f"{op_v:.0f}%"],
        ]
        # 列幅: 25+28+28+28+22+22+27=180mm
        elems.append(_tbl(fin_data,
                          [25*mm, 28*mm, 28*mm, 28*mm, 22*mm, 22*mm, 27*mm]))
        elems.append(Spacer(1, 3*mm))

        # ── 過去株価実績 ──
        try:
            df5 = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if df5 is not None and len(df5) >= 60:
                df5.columns = [c[0] if isinstance(c, tuple) else c for c in df5.columns]
                cl5 = df5["Close"].squeeze()
                cur = float(cl5.iloc[-1])
                ret_parts = []
                for y in [1, 2, 3]:
                    lb = y * 252
                    if len(cl5) > lb:
                        ret = (cur / float(cl5.iloc[-lb]) - 1) * 100
                        ret_parts.append(
                            f"{y}年前比　{'▲' if ret>0 else '▼'}{abs(ret):.0f}%")
                if ret_parts:
                    elems.append(_p(
                        "【過去株価実績】　" + "　　".join(ret_parts), s_body))
        except Exception:
            pass

        # ── リスク要因 ──
        elems += [
            Spacer(1, 3*mm),
            _p("【リスク要因・注意点】",
               _style(f"r_rrh{rank}", fontName=_FONT_BOLD, fontSize=10,
                      textColor=C_RED, spaceAfter=2, leading=14)),
        ]
        for risk in risks:
            elems.append(_p(f"⚠　{risk}", s_risk))

        # ── 投資シナリオ ──
        elems += [
            Spacer(1, 3*mm),
            _p("【投資シナリオ（目安）】",
               _style(f"r_rsh{rank}", fontName=_FONT_BOLD, fontSize=10,
                      textColor=C_ORANGE, spaceAfter=2, leading=14)),
            _p(scenario, s_scenario),
            Spacer(1, 4*mm),
            HRFlowable(width="100%", thickness=0.5,
                       color=colors.HexColor("#aaaaaa"), spaceAfter=5),
        ]

        if rank < len(candidates):
            elems.append(PageBreak())
        time.sleep(0.2)

    # ── 免責事項 ──
    elems += [
        Spacer(1, 6*mm),
        HRFlowable(width="100%", thickness=1, color=C_NAVY, spaceAfter=4),
        _p("【免責事項】",
           _style("r_disc_h", fontName=_FONT_BOLD, fontSize=9,
                  textColor=C_NAVY, spaceAfter=2, leading=14)),
        _p("本レポートは情報提供を目的とした自動生成レポートであり、投資勧誘を意図するものではありません。"
           "投資判断は必ずご自身の責任において行ってください。"
           "株式投資には価格変動リスクが伴い、元本の損失が生じる可能性があります。",
           _style("r_disc", fontSize=8, leading=12, spaceAfter=2,
                  textColor=colors.HexColor("#666666"))),
    ]

    doc.build(elems)
    print(f"[保存] 推奨レポート: {path}")


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
if __name__ == "__main__":
    results_all, stock_data_raw = run_all_screens()

    jp_n = sum(1 for r in results_all if r["市場"] == "JP")
    us_n = len(results_all) - jp_n
    print(f"\nヒット: {len(results_all)}銘柄  (JP:{jp_n} / US:{us_n})")
    for r in results_all[:5]:
        print(f"  {r['市場']} {r['銘柄コード']:8s}  スコア:{r['グロース評価スコア']:5.1f}"
              f"  売上成長:{r.get('売上成長(%)',0):5.0f}%  {r['マッチ戦略'][:60]}")

    save_analysis_pdf(results_all, stock_data_raw)
    save_recommendation_pdf(results_all, stock_data_raw)
