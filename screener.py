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
import io
import json
import shutil
from datetime import datetime

# ── .env 読み込み ──────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── OpenAI ────────────────────────────────
_OPENAI_CLIENT = None
try:
    import openai as _openai_mod
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    if _oai_key:
        _OPENAI_CLIENT = _openai_mod.OpenAI(api_key=_oai_key)
        print("[GPT] OpenAI クライアント初期化完了")
except Exception as _e:
    print(f"[GPT] OpenAI 初期化スキップ: {_e}")

_GPT_CACHE: dict = {}   # キャッシュ（同一データの重複呼び出し防止）

# ── チャート生成 ───────────────────────────
import matplotlib
matplotlib.use("Agg")              # ヘッドレス環境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties, fontManager

# ── 翻訳 ──────────────────────────────────
from deep_translator import GoogleTranslator

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
_FONT_PATH = None  # matplotlib用TTFパス
for _path, _reg, _ in _FONT_CANDIDATES:
    if os.path.exists(_path):
        if _FONT_PATH is None:
            _FONT_PATH = _path
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
    pdfmetrics.registerFontFamily(_reg_normal,
                                  normal=_reg_normal,
                                  bold=_FONT_BOLD,
                                  italic=_reg_normal,
                                  boldItalic=_FONT_BOLD)
else:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))

# matplotlib 日本語フォント設定
_MPL_FP = None
if _FONT_PATH:
    try:
        fontManager.addfont(_FONT_PATH)
        _MPL_FP = FontProperties(fname=_FONT_PATH)
        _mpl_name = _MPL_FP.get_name()
        matplotlib.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": [_mpl_name, "DejaVu Sans"],
            "axes.unicode_minus": False,
        })
    except Exception:
        _MPL_FP = None

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


# 月別マクロ・社会情勢イベント辞書（YYYY-MM → 説明）
_MACRO_EVENTS: dict = {
    "2022-01": "FRBタカ派転換・利上げ開始観測で成長株全般が急落",
    "2022-03": "ロシアによるウクライナ侵攻・資源価格高騰・インフレ加速",
    "2022-05": "FRB 0.5%利上げ・QT開始・インフレピーク懸念",
    "2022-06": "米CPI9.1%・FRB 0.75%利上げ・景気後退懸念が台頭",
    "2022-09": "英国財政危機・ドル独歩高・円安が加速（1ドル=145円超）",
    "2022-10": "FRB利上げ継続・グローバル株安・決算シーズンで業績下方修正相次ぐ",
    "2022-11": "FTX経営破綻・暗号資産市場崩壊・テック株の信用収縮",
    "2022-12": "日銀YCC修正（長期金利上限0.5%引き上げ）・グローバル債券売り",
    "2023-02": "米雇用統計予想大幅超過・利上げ長期化観測で株式市場が一時急落",
    "2023-03": "シリコンバレー銀行（SVB）破綻・米地銀危機・信用不安が世界に波及",
    "2023-05": "米国債務上限交渉が難航・デフォルトリスクでリスクオフ",
    "2023-07": "FRB利上げ0.25%（おそらく最後）・AI関連株への資金集中加速",
    "2023-08": "米国債格下げ（フィッチ）・長期金利急上昇・成長株から資金流出",
    "2023-10": "イスラエル・ハマス紛争勃発・中東地政学リスク・原油高",
    "2023-11": "FRB利上げ停止観測・インフレ鈍化確認で株式市場が急反発",
    "2023-12": "FRB利下げ転換シグナル・年末ラリー・AIブームでナスダック急騰",
    "2024-01": "日本能登半島地震・トランプ返り咲き観測・ドル高継続",
    "2024-03": "日銀マイナス金利解除（初の利上げ）・円高進行・輸出株に売り圧力",
    "2024-04": "中東緊張激化（イラン・イスラエル直接衝突）・原油高・リスクオフ",
    "2024-07": "円キャリートレード急速巻き戻し・急激な円高（145円台→140円台）",
    "2024-08": "日銀追加利上げ（0.25%）・急激な円高・世界株価急落（ブラックマンデー的下落）",
    "2024-09": "FRB 0.5%利下げ開始・米景気軟着陸期待・リスクオン",
    "2024-10": "米大統領選・トランプ再選期待でドル高・金融・エネルギー株上昇",
    "2024-11": "トランプ大統領選勝利・規制緩和期待・テック・金融株が急伸",
    "2024-12": "FRB利下げペース減速示唆・長期金利再上昇・成長株に利食い",
    "2025-01": "トランプ政権始動・関税政策・移民規制が貿易摩擦懸念を喚起",
    "2025-02": "米中関税応酬激化・中国製品に追加関税・サプライチェーン混乱懸念",
    "2025-03": "米国「相互関税」発表・グローバル貿易戦争懸念・リスクオフ全面安",
}

# JP市場：決算集中月（会計年度3月期ベース）
_JP_EARNINGS_SEASON: dict = {
    5:  "3月期本決算シーズン（通期業績・来期見通し発表）",
    8:  "3月期第1四半期決算シーズン",
    11: "3月期第2四半期（中間）決算シーズン",
    2:  "3月期第3四半期決算シーズン",
}
# US市場：四半期決算集中月
_US_EARNINGS_SEASON: dict = {
    1:  "10〜12月期（Q4）決算シーズン",
    4:  "1〜3月期（Q1）決算シーズン",
    7:  "4〜6月期（Q2）決算シーズン",
    10: "7〜9月期（Q3）決算シーズン",
}

# US セクター → ETFシンボル（3段階分解用）
_US_SECTOR_ETF: dict = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Energy":                 "XLE",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
    "Utilities":              "XLU",
    "Semiconductors":         "SOXX",   # 半導体特化
}


def _safe(text: str) -> str:
    """reportlab Paragraph用にHTML特殊文字をエスケープ"""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# テーブルセル用 ParagraphStyle（モジュールレベルで定義）
_cs_hdr   = ParagraphStyle("_csh",  fontName=_FONT_BOLD,   fontSize=8.5,
                             leading=12, textColor=C_WHITE,  wordWrap="CJK")
_cs_body  = ParagraphStyle("_csb",  fontName=_FONT_NORMAL, fontSize=8.5,
                             leading=12, textColor=C_BLACK,  wordWrap="CJK")
_cs_lblue   = ParagraphStyle("_cslb",  fontName=_FONT_BOLD,   fontSize=8.5,
                               leading=12, textColor=C_NAVY,   wordWrap="CJK")
_cs_cat_lbl = ParagraphStyle("_cscat", fontName=_FONT_BOLD,   fontSize=8,
                               leading=11, textColor=C_NAVY,   wordWrap="CJK")
_cs_sum_lbl = ParagraphStyle("_cssum", fontName=_FONT_BOLD,   fontSize=8.5,
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


def _build_move_blocks(sig_moves):
    """
    各変動アノテーションを5カテゴリ構造ブロック（Tableベース）として描画する
    Flowableリストを返す。sig_moves の各要素に "analysis" dict が必要。
    """
    _cat_rows = [
        ("マクロ（市場全体）",       "macro"),
        ("セクター（業界）",         "sector"),
        ("個別企業（ファンダメンタル）", "company"),
        ("需給（資金の流れ）",       "demand"),
        ("テクニカル",               "technical"),
        ("半年後の収益見込み",       "outlook"),
    ]
    elems = []
    for m in sig_moves:
        lbl      = _CIRCLED[m["num"] - 1] if m["num"] <= len(_CIRCLED) else str(m["num"])
        pct_str  = f"{m['pct']:+.1f}%"
        analysis = m.get("analysis", {})

        hdr_text = f"{lbl}　{m['date_str']}　変動率{pct_str}　主要変動要因分析（5カテゴリ）"
        rows = [[Paragraph(_safe(hdr_text), _cs_hdr), ""]]  # SPANされるヘッダー

        for cat_label, key in _cat_rows:
            val = analysis.get(key) or "－"
            rows.append([
                Paragraph(_safe(cat_label), _cs_cat_lbl),
                Paragraph(_safe(val),       _cs_body),
            ])

        short_t = analysis.get("short_term", "")
        mid_t   = analysis.get("mid_term",   "")
        if short_t and short_t != "特定の短期要因なし":
            rows.append([Paragraph("【短期要因】",  _cs_cat_lbl),
                         Paragraph(_safe(short_t), _cs_body)])
        if mid_t and mid_t != "特定の中長期要因なし":
            rows.append([Paragraph("【中長期要因】", _cs_cat_lbl),
                         Paragraph(_safe(mid_t),    _cs_body)])

        summary = analysis.get("summary", "")
        if summary:
            rows.append([Paragraph("■主要因まとめ",  _cs_sum_lbl),
                         Paragraph(_safe(summary),   _cs_sum_lbl)])

        n_rows  = len(rows)
        sum_row = n_rows - 1

        t = Table(rows, colWidths=[40 * mm, 138 * mm])
        t.setStyle(TableStyle([
            ("SPAN",           (0, 0),        (-1, 0),        ),
            ("BACKGROUND",     (0, 0),        (-1, 0),        C_NAVY),
            ("BACKGROUND",     (0, sum_row),  (-1, sum_row),  C_LBLUE),
            ("ROWBACKGROUNDS", (0, 1),        (-1, sum_row - 1), [C_WHITE, C_LGRAY]),
            ("GRID",           (0, 0),        (-1, -1),       0.4, colors.HexColor("#cccccc")),
            ("VALIGN",         (0, 0),        (-1, -1),       "TOP"),
            ("TOPPADDING",     (0, 0),        (-1, -1),       4),
            ("BOTTOMPADDING",  (0, 0),        (-1, -1),       4),
            ("LEFTPADDING",    (0, 0),        (-1, -1),       5),
            ("RIGHTPADDING",   (0, 0),        (-1, -1),       5),
        ]))
        elems.append(t)
        elems.append(Spacer(1, 2 * mm))
    return elems


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
# ─────────────────────────────────────────
# 翻訳ユーティリティ
# ─────────────────────────────────────────
_trans_cache: dict = {}

def _translate(text: str, max_len: int = 400) -> str:
    """英語→日本語翻訳（キャッシュ付き。失敗時は原文返却）"""
    if not text or not text.strip():
        return ""
    text = text[:max_len]
    if text in _trans_cache:
        return _trans_cache[text]
    try:
        result = GoogleTranslator(source="auto", target="ja").translate(text)
        _trans_cache[text] = result or text
    except Exception:
        _trans_cache[text] = text
    return _trans_cache[text]


# ─────────────────────────────────────────
# CSV保存
# ─────────────────────────────────────────
def save_csv(results: list):
    """スクリーニング結果をCSV保存"""
    if not results:
        return

    rows = [{
        "市場":           r["市場"],
        "銘柄コード":     r["銘柄コード"],
        "銘柄名":         r.get("銘柄名", ""),
        "グロース評価スコア": r["グロース評価スコア"],
        "現在値":         r["現在値"],
        "RSI14":          r.get("RSI14", ""),
        "売上成長(%)":    r.get("売上成長(%)", ""),
        "粗利率(%)":      r.get("粗利率(%)", ""),
        "ROE(%)":         r.get("ROE(%)", ""),
        "PER":            r.get("PER", ""),
        "PBR":            r.get("PBR", ""),
        "マッチ戦略数":   r["マッチ戦略数"],
        "マッチ戦略":     r["マッチ戦略"],
        "ハイリスク":     r.get("ハイリスク", False),
    } for r in results]

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "screening_all.csv"),
        index=False, encoding="utf-8-sig")

    # 戦略別サマリー
    strat_rows = []
    for sname in STRATEGIES:
        hits = [r for r in results if sname in r["マッチ戦略"]]
        jp_s = [r["銘柄コード"] for r in hits if r["市場"] == "JP"]
        us_s = [r["銘柄コード"] for r in hits if r["市場"] == "US"]
        strat_rows.append({
            "戦略": sname,
            "国内ヒット数": len(jp_s),
            "米国ヒット数": len(us_s),
            "合計": len(hits),
            "国内銘柄": "、".join(jp_s),
            "米国銘柄": "、".join(us_s),
        })
    pd.DataFrame(strat_rows).to_csv(
        os.path.join(OUTPUT_DIR, "screening_summary.csv"),
        index=False, encoding="utf-8-sig")

    print(f"[保存] CSV: {OUTPUT_DIR}")


# ─────────────────────────────────────────
# チャート生成（変動要因アノテーション付き）
# ─────────────────────────────────────────
_CIRCLED = ["①","②","③","④","⑤","⑥","⑦","⑧"]


def _gpt_move_analysis(ctx: dict) -> "dict | None":
    """
    収集済みコンテキストをGPTに渡して5カテゴリ分析dictを生成する。
    失敗時は None を返す（呼び出し元がローカルロジックにフォールバック）。
    """
    if _OPENAI_CLIENT is None:
        return None

    cache_key = (ctx.get("date_str"), ctx.get("pct"), ctx.get("ticker", ""))
    if cache_key in _GPT_CACHE:
        return _GPT_CACHE[cache_key]

    system_prompt = (
        "あなたは機関投資家向け株式市場の専門アナリストです。\n"
        "与えられた株価変動データを基に、以下の5カテゴリで変動要因を日本語で分析し、"
        "必ず指定のJSON形式のみで返してください。説明文・前置き・コードブロックは不要です。\n\n"
        "出力JSON形式:\n"
        "{\n"
        '  "macro": "マクロ（市場全体）の要因。マクロ経済・金融政策・地政学リスク・市場全体の動向を記載",\n'
        '  "sector": "セクター（業界）の要因。業界トレンド・セクターETF動向・業界特有のイベントを記載",\n'
        '  "company": "個別企業（ファンダメンタル）の要因。決算・業績・製品・経営イベントを記載",\n'
        '  "demand": "需給（資金の流れ）の要因。出来高・機関投資家動向・52週高安値・リスクオン/オフを記載",\n'
        '  "technical": "テクニカルの要因。RSI・ボリンジャーバンド・MACD・SMA乖離・チャートパターンを記載",\n'
        '  "outlook": "半年後の収益見込み。売上・利益成長の方向性・リスク・上値/下値メドを簡潔に記載",\n'
        '  "summary": "最も株価に影響を与えた主要因をカテゴリ名付きで1〜2文でまとめる",\n'
        '  "short_term": "短期要因（数日〜数週間）：ニュース反応・決算・テクニカル過熱など",\n'
        '  "mid_term": "中長期要因（数ヶ月〜）：マクロ政策・業界構造変化・トレンド転換など"\n'
        "}\n\n"
        "【要件】\n"
        "・各カテゴリに具体的な数値・イベント名を含めること\n"
        "・複数要因が重なる場合は優先度を示すこと（例：主因はセクター、副因はマクロ）\n"
        "・推測は「〜とみられる」と明記し、事実と区別すること\n"
        "・各カテゴリ50〜120文字程度を目安に簡潔かつ情報密度高く記載すること\n"
        "・半年後の収益見込みは、当時の業績トレンド・市場環境・セクター動向を踏まえて記載すること"
    )

    data = ctx.copy()
    data.pop("df_ind", None)   # DataFrameは渡さない
    user_prompt = (
        f"以下のデータを分析してください。\n\n"
        f"【基本情報】\n"
        f"- 銘柄: {data.get('ticker','不明')}  市場: {data.get('market','不明')}\n"
        f"- 変動日: {data.get('date_str','不明')}  週次変動率: {data.get('pct',0):+.1f}%\n\n"
        f"【市場・セクターデータ】\n"
        f"- ベンチマーク（{'SPY' if data.get('market')=='US' else '日経225'}）週次: "
        f"{data.get('bm_pct','N/A'):+.1f}%\n"
        f"- セクターETF週次: {data.get('sec_pct','N/A')}"
        + (f"{data.get('sec_pct',0):+.1f}%" if isinstance(data.get('sec_pct'), float) else "N/A")
        + f"\n"
        f"- 市場比セクターα: "
        + (f"{data.get('mkt_alpha',0):+.1f}%" if isinstance(data.get('mkt_alpha'), float) else "N/A")
        + f"\n"
        f"- セクター比個別α: "
        + (f"{data.get('sec_alpha',0):+.1f}%" if isinstance(data.get('sec_alpha'), float) else "N/A")
        + f"\n\n"
        f"【個別企業データ】\n"
        f"- 決算日との関係: {data.get('earnings_tag') or 'なし'}\n"
        f"- 近傍ニュース: {data.get('news_title') or 'なし'}\n\n"
        f"【マクロ・市場環境】\n"
        f"- 当月マクロイベント: {data.get('macro_event') or 'なし'}\n\n"
        f"【テクニカル指標】\n"
        f"- RSI14: {data.get('rsi','N/A')}\n"
        f"- SMA200乖離: {data.get('dev200','N/A')}\n"
        f"- ボリンジャーバンド: 上限={data.get('bb_u','N/A')}, 終値={data.get('close_v','N/A')}, 下限={data.get('bb_l','N/A')}\n"
        f"- MACDヒストグラム: {data.get('macd_h','N/A')}\n"
        f"- 出来高/20日平均: {data.get('vol_ratio',0):.1f}倍\n"
        f"- 52週高値: {data.get('h52','N/A')}, 52週安値: {data.get('l52','N/A')}, 終値: {data.get('close_v','N/A')}\n"
    )

    try:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000,
            timeout=20,
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
        # 必須キーの存在確認
        required = {"macro", "sector", "company", "demand", "technical", "outlook", "summary", "short_term", "mid_term"}
        if not required.issubset(result.keys()):
            return None
        _GPT_CACHE[cache_key] = result
        return result
    except Exception as _e:
        print(f"  [GPT] エラー: {_e}")
        return None


def _detailed_move_analysis(date, pct: float, df_ind: pd.DataFrame,
                             news_list: list, earnings_set: set,
                             market: str = "US",
                             benchmark_wret: "pd.Series | None" = None,
                             sector_wret:    "pd.Series | None" = None,
                             ticker: str = "") -> dict:
    """
    主要変動を5カテゴリで構造化分析したdictを返す。
    keys: macro, sector, company, demand, technical, summary, short_term, mid_term
    """
    ts    = pd.Timestamp(date)
    yr_mo = ts.strftime("%Y-%m")
    month = ts.month

    # ── ベンチマーク・セクターETFのリターン取得 ──
    bm_pct  = None
    sec_pct = None
    if benchmark_wret is not None and len(benchmark_wret) > 0:
        try:
            bi = benchmark_wret.index.get_indexer([ts], method="nearest")[0]
            bm_pct = float(benchmark_wret.iloc[bi]) * 100
        except Exception:
            pass
    if sector_wret is not None and len(sector_wret) > 0:
        try:
            si = sector_wret.index.get_indexer([ts], method="nearest")[0]
            sec_pct = float(sector_wret.iloc[si]) * 100
        except Exception:
            pass

    mkt_alpha = (sec_pct - bm_pct) if (bm_pct is not None and sec_pct is not None) else None
    sec_alpha = (pct - sec_pct)     if sec_pct is not None else (
                (pct - bm_pct)      if bm_pct is not None else None)

    # ── テクニカル指標の取得 ──
    tech = {}
    try:
        idx = df_ind.index.get_indexer([date], method="nearest")[0]
        row = df_ind.iloc[idx]
        tech["close"]  = float(row.get("Close")     or 0)
        tech["h52"]    = float(row.get("HIGH_52W")  or np.nan)
        tech["l52"]    = float(row.get("LOW_52W")   or np.nan)
        tech["vol"]    = float(row.get("Volume")    or 0)
        tech["vol_ma"] = float(row.get("VOL_MA20")  or 0)
        tech["rsi"]    = float(row.get("RSI14")     or np.nan)
        tech["dev200"] = float(row.get("DEV_SMA200") or np.nan)
        tech["bb_u"]   = float(row.get("BB_upper")  or 0)
        tech["bb_l"]   = float(row.get("BB_lower")  or 0)
        tech["macd_h"] = float(row.get("MACD_hist") or np.nan)
    except Exception:
        pass

    vol_ratio = (tech.get("vol", 0) / tech["vol_ma"]
                 if tech.get("vol_ma", 0) > 0 else 0)
    close_v = tech.get("close", 0)

    # ── 決算日前後チェック ──
    earnings_tag = None
    move_date = ts.normalize()
    for d in range(-3, 8):
        chk = move_date + pd.Timedelta(days=d)
        if chk in earnings_set:
            earnings_tag = ("発表3日前" if d == -3 else "発表2日前" if d == -2 else
                            "発表前日"  if d == -1 else "発表当日"  if d == 0 else
                            f"発表後{d}日")
            break

    # ── 近傍ニュース取得（±7日以内） ──
    target_ts = ts.timestamp()
    near_news = sorted(
        [(abs(n.get("providerPublishTime", 0) - target_ts),
          n.get("title") or (n.get("content") or {}).get("title") or "")
         for n in news_list
         if abs(n.get("providerPublishTime", 0) - target_ts) <= 7 * 86400
         and (n.get("title") or (n.get("content") or {}).get("title") or "")]
    )
    news_title = _translate(near_news[0][1], 90) if near_news else None

    macro_event = _MACRO_EVENTS.get(yr_mo)

    # ── GPT分析を優先試行（失敗時はローカルロジックにフォールバック） ──
    _gpt_ctx = {
        "ticker":       ticker,
        "date_str":     ts.strftime("%Y/%m/%d"),
        "market":       market,
        "pct":          pct,
        "bm_pct":       bm_pct,
        "sec_pct":      sec_pct,
        "mkt_alpha":    mkt_alpha,
        "sec_alpha":    sec_alpha,
        "earnings_tag": earnings_tag,
        "news_title":   news_title,
        "macro_event":  macro_event,
        "rsi":          tech.get("rsi"),
        "dev200":       tech.get("dev200"),
        "bb_u":         tech.get("bb_u"),
        "bb_l":         tech.get("bb_l"),
        "close_v":      close_v,
        "macd_h":       tech.get("macd_h"),
        "vol_ratio":    vol_ratio,
        "h52":          tech.get("h52"),
        "l52":          tech.get("l52"),
    }
    _gpt_result = _gpt_move_analysis(_gpt_ctx)
    if _gpt_result is not None:
        return _gpt_result

    # ─── ① マクロ（市場全体） ───────────────────────────
    macro_parts = []
    if macro_event:
        macro_parts.append(macro_event)
    if bm_pct is not None:
        bm_sym = "SPY（S&P500）" if market == "US" else "日経225"
        if abs(bm_pct) >= 3.5:
            side = "大幅上昇" if bm_pct > 0 else "大幅下落"
            macro_parts.append(f"市場全体も{side}（{bm_sym} {bm_pct:+.1f}%）")
        elif abs(bm_pct) >= 1.5:
            side = "上昇" if bm_pct > 0 else "下落"
            macro_parts.append(f"市場は{side}基調（{bm_sym} {bm_pct:+.1f}%）")
        else:
            macro_parts.append(f"市場はほぼ横ばい（{bm_sym} {bm_pct:+.1f}%）")
    if not macro_event:
        if market == "JP":
            season = _JP_EARNINGS_SEASON.get(month)
            if season:
                macro_parts.append(f"時期：{season}")
        else:
            season = _US_EARNINGS_SEASON.get(month)
            if season:
                macro_parts.append(f"時期：{season}")
    macro_str = "　".join(macro_parts) if macro_parts else "特定のマクロイベントなし"

    # ─── ② セクター（業界） ─────────────────────────────
    sector_parts = []
    if sec_pct is not None and bm_pct is not None:
        sector_parts.append(f"セクターETF {sec_pct:+.1f}%（市場比α {mkt_alpha:+.1f}%）")
        if abs(mkt_alpha) >= 4:
            ind = "大幅アウトパフォーム" if mkt_alpha > 0 else "大幅アンダーパフォーム"
            sector_parts.append(f"セクター全体が市場を{ind}")
        elif abs(mkt_alpha) >= 2:
            ind = "アウトパフォーム" if mkt_alpha > 0 else "アンダーパフォーム"
            sector_parts.append(f"セクターが市場を{ind}")
    elif sec_pct is not None:
        sector_parts.append(f"セクターETF {sec_pct:+.1f}%")
    if market == "JP":
        season = _JP_EARNINGS_SEASON.get(month)
        if season:
            sector_parts.append(f"時期：{season}")
    else:
        season = _US_EARNINGS_SEASON.get(month)
        if season:
            sector_parts.append(f"時期：{season}")
    sector_str = "　".join(sector_parts) if sector_parts else "セクターETFデータなし（JP市場または取得不可）"

    # ─── ③ 個別企業（ファンダメンタル） ─────────────────
    company_parts = []
    if earnings_tag:
        direction = "上方修正・好決算への期待／反応" if pct > 0 else "下方修正・業績悪化への懸念／反応"
        company_parts.append(f"決算{earnings_tag}：{direction}")
    if news_title:
        company_parts.append(f"直近ニュース：{news_title}")
    if sec_alpha is not None and abs(sec_alpha) >= 5:
        base_label = "セクター対比" if sec_pct is not None else "市場対比"
        ind = "アウトパフォーム" if sec_alpha > 0 else "アンダーパフォーム"
        company_parts.append(f"{base_label}個別α {sec_alpha:+.1f}%（強い個別株{ind}）")
    company_str = "　".join(company_parts) if company_parts else "特定の個別企業材料なし（セクター・市場連動とみられる）"

    # ─── ④ 需給（資金の流れ） ────────────────────────────
    demand_parts = []
    if vol_ratio >= 3.0:
        demand_parts.append(f"出来高20日平均比{vol_ratio:.1f}倍に急増（機関投資家の大口売買を示唆）")
    elif vol_ratio >= 1.8:
        demand_parts.append(f"出来高増加（平均比{vol_ratio:.1f}倍）")
    elif 0 < vol_ratio <= 0.5:
        demand_parts.append("出来高閑散（流動性薄く値動きが誇張された可能性）")

    h52 = tech.get("h52", np.nan)
    l52 = tech.get("l52", np.nan)
    if close_v > 0 and not np.isnan(h52) and h52 > 0:
        if close_v >= h52:
            demand_parts.append("52週高値を更新（新高値ブレイクアウト・買いモメンタム加速）")
        elif close_v >= h52 * 0.97:
            demand_parts.append(f"52週高値（{h52:.1f}）付近での高値圏攻防")
    if close_v > 0 and not np.isnan(l52) and l52 > 0:
        if close_v <= l52:
            demand_parts.append("52週安値を更新（下値サポート喪失・売り継続リスク）")
        elif close_v <= l52 * 1.05:
            demand_parts.append(f"52週安値（{l52:.1f}）付近での底値圏攻防")

    if bm_pct is not None and abs(bm_pct) >= 3:
        flow = "リスクオン（市場全体への資金流入局面）" if bm_pct > 0 else "リスクオフ（市場全体からの資金流出局面）"
        demand_parts.append(flow)

    if not demand_parts:
        direction = "買い需要が優勢" if pct > 0 else "売り圧力が優勢"
        demand_parts.append(f"特段の需給異常なし（{direction}）")
    demand_str = "　".join(demand_parts)

    # ─── ⑤ テクニカル ─────────────────────────────────
    tech_parts = []
    rsi = tech.get("rsi", np.nan)
    if not np.isnan(rsi):
        if rsi >= 80:
            tech_parts.append(f"RSI {rsi:.0f}（極端な過買われ・短期過熱）")
        elif rsi >= 70:
            tech_parts.append(f"RSI {rsi:.0f}（過買われ域・利食い圧力）")
        elif rsi <= 22:
            tech_parts.append(f"RSI {rsi:.0f}（極端な売られ過ぎ・自律反発の可能性）")
        elif rsi <= 32:
            tech_parts.append(f"RSI {rsi:.0f}（売られ過ぎ・底打ち模索）")
        else:
            tech_parts.append(f"RSI {rsi:.0f}（中立域）")

    bb_u = tech.get("bb_u", 0)
    bb_l = tech.get("bb_l", 0)
    if close_v > 0 and bb_u > 0:
        if close_v > bb_u:
            tech_parts.append("ボリンジャーバンド+2σ突破（強いブレイクアウトシグナル）")
        elif bb_l > 0 and close_v < bb_l:
            tech_parts.append("ボリンジャーバンド-2σ割れ（強い売りシグナル）")

    dev200 = tech.get("dev200", np.nan)
    if not np.isnan(dev200):
        if abs(dev200) >= 40:
            side = "上方" if dev200 > 0 else "下方"
            tech_parts.append(f"SMA200から{abs(dev200):.0f}%{side}乖離（過熱/叩き売り圏）")
        elif abs(dev200) >= 15:
            side = "上" if dev200 > 0 else "下"
            tech_parts.append(f"SMA200から{abs(dev200):.0f}%乖離（{side}方向への強いトレンド）")
        elif abs(dev200) <= 3:
            tech_parts.append(f"SMA200付近（±{abs(dev200):.0f}%）での攻防")

    macd_h = tech.get("macd_h", np.nan)
    if not np.isnan(macd_h):
        if macd_h > 0 and pct > 0:
            tech_parts.append("MACDヒストグラム拡大（上昇モメンタム継続）")
        elif macd_h < 0 and pct < 0:
            tech_parts.append("MACDヒストグラム縮小（下落モメンタム継続）")
        elif macd_h > 0 and pct < 0:
            tech_parts.append("MACDはポジティブも株価は反落（上昇一服・反落局面）")
        elif macd_h < 0 and pct > 0:
            tech_parts.append("MACDはネガティブも株価は反発（自律反発・逆張り動意）")
    tech_str = "　".join(tech_parts) if tech_parts else "特段のテクニカルシグナルなし"

    # ─── 主要因の優先度判定（複数要因が重なる場合） ────────
    if earnings_tag:
        priority_cat    = "個別企業（ファンダメンタル）"
        summary = f"{priority_cat}が主因：決算{earnings_tag}への反応"
        if news_title:
            summary += f"　関連ニュース：{news_title[:30]}"
    elif sec_alpha is not None and abs(sec_alpha) >= 7:
        priority_cat = "③個別企業（ファンダメンタル）"
        base = "セクター対比" if sec_pct is not None else "市場対比"
        ind  = "アウトパフォーム" if sec_alpha > 0 else "アンダーパフォーム"
        summary = f"{priority_cat}が主因：{base}α {sec_alpha:+.1f}%の強い個別動き"
        if news_title:
            summary += f"（{news_title[:30]}）"
    elif news_title and (bm_pct is None or abs(pct - (bm_pct or 0)) >= 5):
        priority_cat = "③個別企業（ファンダメンタル）"
        summary = f"{priority_cat}が主因：{news_title[:40]}"
    elif mkt_alpha is not None and abs(mkt_alpha) >= 4:
        priority_cat = "セクター（業界）"
        ind  = "アウトパフォーム" if mkt_alpha > 0 else "アンダーパフォーム"
        summary = f"{priority_cat}が主因：セクターETFが市場比α {mkt_alpha:+.1f}%で{ind}"
    elif bm_pct is not None and abs(bm_pct) >= 3.5 and (sec_alpha is None or abs(sec_alpha) < 5):
        priority_cat = "マクロ（市場全体）"
        side = "上昇" if bm_pct > 0 else "下落"
        summary = f"{priority_cat}が主因：市場全体の{side}に連動（指数 {bm_pct:+.1f}%）"
        if macro_event:
            summary += f"　背景：{macro_event[:30]}"
    elif macro_event and not earnings_tag and not news_title:
        priority_cat = "マクロ（市場全体）"
        summary = f"{priority_cat}が主因：{macro_event[:40]}"
    elif vol_ratio >= 2.5:
        priority_cat = "需給（資金の流れ）"
        summary = f"{priority_cat}が主因：出来高 {vol_ratio:.1f}倍急増による需給インパクト"
    else:
        priority_cat = "テクニカル"
        summary = f"{priority_cat}が主因：明確なファンダ材料なし、テクニカル主導の値動き"

    # ─── 短期 / 中長期要因の区別 ───────────────────────
    short_factors = []
    mid_factors   = []
    if earnings_tag:
        short_factors.append(f"決算{earnings_tag}への即時市場反応")
    if news_title:
        short_factors.append("直近ニュースへの短期的な反応")
    if not np.isnan(rsi) and (rsi >= 70 or rsi <= 32):
        short_factors.append("テクニカル過熱／過売りによる短期需給変動")
    if vol_ratio >= 2.0:
        short_factors.append(f"大口売買（出来高{vol_ratio:.1f}倍）による短期的な需給インパクト")

    if macro_event:
        mid_factors.append(f"マクロ政策・情勢の継続的影響（{macro_event[:25]}）")
    if mkt_alpha is not None and abs(mkt_alpha) >= 2:
        mid_factors.append("セクターローテーション・業界トレンドの変化")
    if not np.isnan(dev200) and abs(dev200) >= 15:
        side = "上昇" if dev200 > 0 else "下落"
        mid_factors.append(f"SMA200との乖離が拡大（中長期的な{side}トレンドの継続）")

    short_str = "　".join(short_factors) if short_factors else "特定の短期要因なし"
    mid_str   = "　".join(mid_factors)   if mid_factors   else "特定の中長期要因なし"

    # ── 半年後の収益見込み（ローカルロジック簡易推定） ──
    outlook_parts = []
    if macro_event:
        outlook_parts.append(f"マクロ環境（{macro_event[:20]}）の影響が継続する見込み")
    if mkt_alpha is not None and abs(mkt_alpha) >= 2:
        ind = "追い風" if mkt_alpha > 0 else "向かい風"
        outlook_parts.append(f"セクタートレンドは{ind}（ETF市場比α {mkt_alpha:+.1f}%）")
    if not np.isnan(dev200) and dev200 > 15:
        outlook_parts.append(f"SMA200を{dev200:.0f}%上回る強いトレンド継続中")
    elif not np.isnan(dev200) and dev200 < -15:
        outlook_parts.append(f"SMA200を{abs(dev200):.0f}%下回り回復には時間を要する見込み")
    if not outlook_parts:
        direction = "緩やかな上昇" if pct > 0 else "横ばい〜回復"
        outlook_parts.append(f"現状トレンドが継続すれば{direction}が見込まれる")
    outlook_str = "　".join(outlook_parts)

    return {
        "macro":      macro_str,
        "sector":     sector_str,
        "company":    company_str,
        "demand":     demand_str,
        "technical":  tech_str,
        "outlook":    outlook_str,
        "summary":    summary,
        "short_term": short_str,
        "mid_term":   mid_str,
    }


def generate_chart_with_annotations(
        ticker: str, market: str,
        df_raw: pd.DataFrame, earnings_set: set = None,
        benchmark_wret: "pd.Series | None" = None,
        sector_wret:    "pd.Series | None" = None,
) -> tuple:
    """
    変動要因番号アノテーション付き株価チャートを生成。
    Returns: (io.BytesIO | None, annotations: list[dict])
    """
    if df_raw is None or len(df_raw) < 30:
        return None, []

    try:
        df_ind = calc_indicators(df_raw.copy())

        if isinstance(df_ind.columns, pd.MultiIndex):
            df_ind.columns = [c[0] for c in df_ind.columns]
        df_ind.index = pd.to_datetime(df_ind.index)
        df_ind = df_ind.sort_index()

        close  = df_ind["Close"].squeeze().dropna()
        volume = df_ind["Volume"].squeeze().fillna(0)

        p = MARKET_PARAMS.get(market, MARKET_PARAMS["US"])

        def sma(n):
            return close.rolling(n).mean()

        sma_s = sma(int(p["sma_s"].replace("SMA", "")))
        sma_m = sma(int(p["sma_m"].replace("SMA", "")))
        sma_l = sma(200)

        # ── 主要変動を週次リターンから抽出 ──
        weekly = close.resample("W-FRI").last().dropna()
        wret   = weekly.pct_change().dropna()
        sig_w  = wret[wret.abs() >= 0.08]
        if len(sig_w) > 8:
            sig_w = sig_w.loc[sig_w.abs().nlargest(8).index].sort_index()

        # ニュース取得
        try:
            news_list = yf.Ticker(ticker).news or []
        except Exception:
            news_list = []

        sig_moves = []
        for w_date, w_pct in sig_w.items():
            idx = close.index.get_indexer([w_date], method="nearest")[0]
            d_date  = close.index[idx]
            d_price = float(close.iloc[idx])
            analysis = _detailed_move_analysis(d_date, w_pct * 100, df_ind, news_list,
                                               earnings_set or set(), market,
                                               benchmark_wret, sector_wret,
                                               ticker=ticker)
            sig_moves.append({
                "num":      len(sig_moves) + 1,
                "date":     d_date,
                "date_str": d_date.strftime("%Y/%m/%d"),
                "price":    d_price,
                "pct":      w_pct * 100,
                "analysis": analysis,
            })

        # ── チャート描画 ──
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6.5),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True)
        fig.patch.set_facecolor("white")

        kw_fp = {"fontproperties": _MPL_FP} if _MPL_FP else {}

        ax1.plot(close.index, close.values,
                 color="#1a3a5c", linewidth=1.4, label="終値", zorder=3)
        ax1.plot(sma_s.index, sma_s.values,
                 color="#2563a8", linewidth=0.9, linestyle="--",
                 alpha=0.85, label=p["sma_s"])
        ax1.plot(sma_m.index, sma_m.values,
                 color="#f57c00", linewidth=0.9, linestyle="--",
                 alpha=0.85, label=p["sma_m"])
        ax1.plot(sma_l.index, sma_l.values,
                 color="#dc2626", linewidth=0.9, linestyle="--",
                 alpha=0.85, label="SMA200")

        # Bollinger Bands shading (after SMA lines)
        if "BB_upper" in df_ind.columns and "BB_lower" in df_ind.columns:
            ax1.fill_between(df_ind.index, df_ind["BB_upper"], df_ind["BB_lower"],
                             alpha=0.06, color="#2563a8", label="BB(±2σ)")

        # アノテーション（番号付き丸）
        y_min = float(close.min())
        y_max = float(close.max())
        y_rng = max(y_max - y_min, y_max * 0.01)
        for i, m in enumerate(sig_moves):
            fc = "#15803d" if m["pct"] > 0 else "#dc2626"
            offset = y_rng * (0.14 if i % 2 == 0 else -0.14)
            ann_y  = np.clip(m["price"] + offset, y_min - y_rng * 0.05,
                             y_max + y_rng * 0.05)
            lbl = _CIRCLED[i] if i < len(_CIRCLED) else str(m["num"])
            ax1.annotate(
                lbl,
                xy=(m["date"], m["price"]),
                xytext=(m["date"], ann_y),
                fontsize=9, color="white",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.22",
                          fc=fc, ec="white", lw=0.5),
                arrowprops=dict(arrowstyle="-",
                                color=fc, lw=0.7),
                **kw_fp,
            )

        ax1.set_ylabel("株価", fontsize=9, **kw_fp)
        ax1.legend(loc="upper left", fontsize=7, ncol=4,
                   prop=_MPL_FP if _MPL_FP else None)
        currency_unit = "（円）" if market == "JP" else "（USD）"
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: (f"{x/10000:.1f}万" if x >= 10000 else f"{x:,.0f}")
                if market == "JP" else f"{x:,.1f}"
            ))
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.set_facecolor("#f8f9fa")

        # タイトル（番号の凡例を添える）
        title_base = f"{ticker}  株価チャート（過去2年）  {currency_unit}"
        ax1.set_title(title_base, fontsize=10, fontweight="bold",
                      color="#1a3a5c", pad=6, **kw_fp)

        # 出来高
        c_vol = ["#15803d" if (i == 0 or close.iloc[i] >= close.iloc[i-1])
                 else "#dc2626"
                 for i in range(len(close))]
        ax2.bar(volume.index, volume.values,
                color=c_vol, alpha=0.65, width=1)
        ax2.set_ylabel("出来高", fontsize=9, **kw_fp)
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: (f"{x/1e4:.0f}万" if x >= 1e4 else f"{x:.0f}")
                if market == "JP"
                else (f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
            ))
        ax2.grid(True, alpha=0.3, linewidth=0.5, axis="y")
        ax2.set_facecolor("#f8f9fa")

        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        plt.tight_layout(pad=1.0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf, sig_moves

    except Exception as e:
        print(f"  [チャート生成エラー] {ticker}: {e}")
        return None, []


# ─────────────────────────────────────────
# PDF出力: 統合レポート（推奨銘柄＋全体分析）
# ─────────────────────────────────────────
def save_report_pdf(results: list, stock_data: dict):
    """推奨銘柄レポートと全体分析を1つのPDFに統合して出力"""
    if not results:
        print("ヒット銘柄なし"); return

    path = os.path.join(OUTPUT_DIR, "screening_report.pdf")
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)

    # ── 共通スタイル ──
    s_title    = _style("s_title",  fontSize=22, fontName=_FONT_BOLD,
                        textColor=C_NAVY, spaceAfter=6, leading=30)
    s_part     = _style("s_part",   fontSize=16, fontName=_FONT_BOLD,
                        textColor=C_WHITE, spaceAfter=0, leading=22)
    s_sub      = _style("s_sub",    fontSize=12, fontName=_FONT_BOLD,
                        textColor=C_BLUE, spaceAfter=4, leading=18)
    s_h3       = _style("s_h3",     fontSize=11, fontName=_FONT_BOLD,
                        textColor=C_NAVY, spaceAfter=2, leading=16)
    s_body     = _style("s_body",   fontSize=9,  leading=14, spaceAfter=3)
    s_small    = _style("s_small",  fontSize=8,  leading=12, spaceAfter=2,
                        textColor=colors.HexColor("#444444"))
    s_green    = _style("s_green",  fontSize=9.5, leading=14, leftIndent=10,
                        spaceAfter=3, textColor=C_GREEN)
    s_risk     = _style("s_risk",   fontSize=9,  leading=13, leftIndent=10,
                        spaceAfter=2, textColor=C_RED)
    s_scenario = _style("s_scen",   fontSize=10, fontName=_FONT_BOLD, leading=16,
                        textColor=C_ORANGE, spaceAfter=4)
    s_match    = _style("s_match",  fontSize=8.5, leading=13, spaceAfter=3,
                        textColor=colors.HexColor("#333333"))

    jp_hits = [r for r in results if r["市場"] == "JP"]
    us_hits = [r for r in results if r["市場"] == "US"]
    top     = [r for r in results if r["グロース評価スコア"] >= 5]
    candidates = sorted(
        [r for r in results if not r.get("ハイリスク") and r["マッチ戦略数"] >= 2],
        key=lambda x: -x["グロース評価スコア"])[:10]

    # ── ベンチマーク週次リターン（市場連動判定用） ──────────
    print("ベンチマーク・セクターETF取得中...")
    _bm_wret: dict = {}
    for _mkt, _bm_sym in [("US", "SPY"), ("JP", "1321.T")]:
        try:
            _df_bm = yf.download(_bm_sym, period="2y", progress=False,
                                 auto_adjust=True)
            if _df_bm is not None and len(_df_bm) > 10:
                _df_bm.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in _df_bm.columns]
                _bm_wret[_mkt] = (_df_bm["Close"]
                                  .resample("W-FRI").last()
                                  .pct_change().dropna())
        except Exception:
            pass

    # ── US セクターETF週次リターン（3段階分解用）──────────
    # 対象銘柄のセクターを収集して必要分だけ取得
    _sec_etf_wret: dict = {}   # sector_name → pd.Series
    _secs_needed = set()
    for _r in results:
        if _r["市場"] == "US":
            _s = (_r.get("_info") or {}).get("sector", "")
            if _s in _US_SECTOR_ETF:
                _secs_needed.add(_s)
    for _sec_name in _secs_needed:
        _etf_sym = _US_SECTOR_ETF[_sec_name]
        try:
            _df_etf = yf.download(_etf_sym, period="2y", progress=False,
                                  auto_adjust=True)
            if _df_etf is not None and len(_df_etf) > 10:
                _df_etf.columns = [c[0] if isinstance(c, tuple) else c
                                   for c in _df_etf.columns]
                _sec_etf_wret[_sec_name] = (_df_etf["Close"]
                                            .resample("W-FRI").last()
                                            .pct_change().dropna())
                print(f"  セクターETF取得: {_etf_sym} ({_sec_name})")
        except Exception:
            pass

    elems = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 表紙
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elems += [
        Spacer(1, 16*mm),
        _p("株式スクリーニング　総合レポート", s_title),
        _p("グロース重視スクリーニング　全27戦略", s_sub),
        _p(f"実行日: {TODAY}　　対象: 国内株式（東証）/ 米国株式（NYSE・NASDAQ）", s_body),
        HRFlowable(width="100%", thickness=2.5, color=C_NAVY, spaceAfter=8),
        Spacer(1, 5*mm),
    ]

    sum_data = [
        ["項目", "国内株", "米国株", "合計"],
        ["スキャン銘柄数", f"{_n_jp}銘柄", f"{_n_us}銘柄", f"{_n_jp+_n_us}銘柄"],
        ["ヒット銘柄数",   f"{len(jp_hits)}銘柄", f"{len(us_hits)}銘柄", f"{len(results)}銘柄"],
        ["グロース評価スコア5以上", "－", "－", f"{len(top)}銘柄"],
        ["推奨銘柄数（ハイリスク除外・2戦略以上）", "－", "－", f"{len(candidates)}銘柄"],
    ]
    elems.append(_tbl(sum_data, [95*mm, 28*mm, 28*mm, 29*mm]))
    elems.append(Spacer(1, 6*mm))

    # 目次
    elems += [
        _p("【本レポートの構成】", _style("s_toc_h", fontName=_FONT_BOLD,
           fontSize=10, leading=15, spaceAfter=3, textColor=C_NAVY)),
        _p(f"　PART 1　今買うべき推奨銘柄　（{len(candidates)}銘柄 / 各銘柄チャート・変動分析・投資シナリオ付き）", s_body),
        _p("　PART 2　全体スクリーニング分析　（戦略別ヒット数・全銘柄一覧・注目銘柄詳細）", s_body),
        Spacer(1, 4*mm),
    ]
    elems.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 1: 推奨銘柄
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    part1_hdr = Table(
        [[Paragraph("PART 1　今買うべき推奨銘柄", s_part)]],
        colWidths=[180*mm])
    part1_hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_NAVY),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ]))
    elems += [
        part1_hdr,
        Spacer(1, 5*mm),
        _p("グロース重視 × ハイリスク除外 × 中期投資", s_sub),
        Spacer(1, 3*mm),
        _p("【選定基準】", _style("s_crit_h", fontName=_FONT_BOLD, fontSize=9.5,
           textColor=C_NAVY, spaceAfter=2, leading=14)),
        _p("・ 27戦略スクリーニングで2戦略以上にヒット", s_body),
        _p("・ グロース評価スコア上位（売上成長・粗利率・ROE・モメンタム総合評価）", s_body),
        _p("・ ハイリスク除外（RSI75超の過熱 / SMA200から60%超乖離 / 年率ボラティリティ85%超）", s_body),
        _p("・ 推奨順位はグロース評価スコア降順", s_body),
        Spacer(1, 4*mm),
    ]

    # 推奨銘柄概要テーブル
    ov_data = [["順位", "市場", "コード", "銘柄名", "スコア",
                "現在値", "売上成長%", "粗利率%", "ROE%", "マッチ戦略数"]]
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
            str(r["マッチ戦略数"]),
        ])
    elems.append(_tbl(ov_data, [10*mm, 12*mm, 18*mm, 42*mm, 14*mm,
                                 24*mm, 18*mm, 16*mm, 14*mm, 12*mm]))
    elems.append(PageBreak())

    # ── 各推奨銘柄詳細 ──
    print(f"\n[PART1] 推奨銘柄詳細生成中... ({len(candidates)}銘柄)")
    for rank, r in enumerate(candidates, 1):
        ticker = r["銘柄コード"]
        market = r["市場"]
        info   = r.get("_info") or fetch_info(ticker)
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

        # 決算日取得
        try:
            ed_df = yf.Ticker(ticker).earnings_dates
            if ed_df is not None and len(ed_df) > 0:
                idx_ed = ed_df.index
                idx_ed = idx_ed.tz_localize(None) if idx_ed.tz is None else idx_ed.tz_convert(None)
                earnings_set = set(pd.DatetimeIndex(idx_ed).normalize().tolist())
            else:
                earnings_set = set()
        except Exception:
            earnings_set = set()

        raw_df     = stock_data.get(ticker)
        _sec_raw   = (info.get("sector") or "") if market == "US" else ""
        img_buf, sig_moves = generate_chart_with_annotations(
            ticker, market, raw_df, earnings_set,
            _bm_wret.get(market), _sec_etf_wret.get(_sec_raw))

        # 推奨理由・リスク・シナリオ
        df_ind = pd.DataFrame()
        if raw_df is not None:
            try:
                df_ind = calc_indicators(raw_df.copy())
            except Exception:
                pass
        reasons, risks, scenario = generate_buy_reasons(r, df_ind, info, market)

        # ── 銘柄ヘッダー ──
        hd_s = _style(f"p1_hd{rank}", fontSize=13, fontName=_FONT_BOLD,
                      textColor=C_NAVY, spaceAfter=2, leading=19)
        sm_s = _style(f"p1_sm{rank}", fontSize=8.5, leading=13, spaceAfter=1,
                      textColor=colors.HexColor("#444444"))
        elems += [
            _p(f"推奨 #{rank}　{ticker}　{name}", hd_s),
            _p(f"市場: {'国内株（東証）' if market=='JP' else '米国株（NYSE/NASDAQ）'}　"
               f"セクター: {sector}　"
               f"現在値: {r['現在値']}{currency}　"
               f"グロース評価スコア: {r['グロース評価スコア']}", sm_s),
            HRFlowable(width="100%", thickness=1.5, color=C_ORANGE, spaceAfter=5),
        ]

        # 今買うべき理由
        elems.append(_p("【今買うべき理由】",
                        _style(f"p1_rh{rank}", fontName=_FONT_BOLD, fontSize=10.5,
                               textColor=C_GREEN, spaceAfter=2, leading=15)))
        for reason in reasons:
            elems.append(_p(f"✓　{reason}", s_green))
        elems.append(Spacer(1, 3*mm))

        # マッチ戦略
        elems += [
            _p("【マッチした戦略シグナル】",
               _style(f"p1_mh{rank}", fontName=_FONT_BOLD, fontSize=9.5,
                      textColor=C_BLUE, spaceAfter=2, leading=14)),
            _p(r["マッチ戦略"].replace(" | ", "　/　"), s_match),
            Spacer(1, 3*mm),
        ]

        # 財務指標テーブル
        fin_data = [
            ["時価総額", "PER", "PBR", "ROE（自己資本利益率）",
             "売上成長率", "粗利率", "営業利益率"],
            [mktcap_s,
             f"{per_v:.0f}倍" if per_v else "N/A",
             f"{pbr_v:.1f}倍" if pbr_v else "N/A",
             f"{roe_v:.0f}%",
             f"{rev_v:.0f}%",
             f"{gross_v:.0f}%",
             f"{op_v:.0f}%"],
        ]
        elems.append(_tbl(fin_data,
                          [26*mm, 18*mm, 18*mm, 36*mm, 24*mm, 20*mm, 38*mm]))
        elems.append(Spacer(1, 3*mm))

        # チャート
        if img_buf:
            from reportlab.platypus import Image as RLImage
            elems += [
                _p("【株価チャートと主要変動分析】",
                   _style(f"p1_chh{rank}", fontName=_FONT_BOLD, fontSize=10,
                          textColor=C_BLUE, spaceAfter=2, leading=14)),
                RLImage(img_buf, width=178*mm, height=88*mm),
                Spacer(1, 1*mm),
            ]
            if sig_moves:
                elems += _build_move_blocks(sig_moves)
        elems.append(Spacer(1, 3*mm))

        # リスク要因
        elems += [
            _p("【リスク要因・注意点】",
               _style(f"p1_rrh{rank}", fontName=_FONT_BOLD, fontSize=10,
                      textColor=C_RED, spaceAfter=2, leading=14)),
        ]
        for risk in risks:
            elems.append(_p(f"⚠　{risk}", s_risk))

        # 投資シナリオ
        elems += [
            Spacer(1, 3*mm),
            _p("【投資シナリオ（目安）】",
               _style(f"p1_rsh{rank}", fontName=_FONT_BOLD, fontSize=10,
                      textColor=C_ORANGE, spaceAfter=2, leading=14)),
            _p(scenario, s_scenario),
            Spacer(1, 4*mm),
            HRFlowable(width="100%", thickness=0.5,
                       color=colors.HexColor("#aaaaaa"), spaceAfter=4),
        ]

        if rank < len(candidates):
            elems.append(PageBreak())
        time.sleep(0.2)

    elems.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 2: スクリーニング全体分析
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    part2_hdr = Table(
        [[Paragraph("PART 2　スクリーニング全体分析", s_part)]],
        colWidths=[180*mm])
    part2_hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_NAVY),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ]))
    elems += [
        part2_hdr,
        Spacer(1, 5*mm),
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
    elems.append(_tbl(st_data, [56*mm, 10*mm, 10*mm, 10*mm, 47*mm, 47*mm]))
    elems.append(PageBreak())

    # 全ヒット銘柄一覧
    elems += [
        _p("■ ヒット銘柄一覧（グロース評価スコア順）", s_sub),
        HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=4),
        _p("※ リスク欄: ○=通常　⚠高=高リスク（RSI過熱・大幅乖離・高ボラ）", s_small),
        Spacer(1, 2*mm),
    ]
    all_data = [["市場", "コード", "銘柄名", "スコア", "現在値",
                 "RSI", "売上成長%", "粗利率%", "ROE%", "マッチ数", "リスク"]]
    for r in results:
        name_s   = (r.get("銘柄名") or r["銘柄コード"])[:16]
        risk_str = "⚠高" if r.get("ハイリスク") else "○"
        all_data.append([
            r["市場"], r["銘柄コード"], name_s,
            str(r["グロース評価スコア"]), str(r["現在値"]),
            str(r.get("RSI14") or "－"),
            f"{r.get('売上成長(%)',0):.0f}%",
            f"{r.get('粗利率(%)',0):.0f}%",
            f"{r.get('ROE(%)',0):.0f}%",
            str(r["マッチ戦略数"]), risk_str,
        ])
    elems.append(_tbl(all_data,
                      [10*mm, 18*mm, 40*mm, 13*mm, 20*mm,
                       10*mm, 17*mm, 14*mm, 12*mm, 14*mm, 12*mm]))
    elems.append(PageBreak())

    # 注目銘柄 詳細分析（スコア4以上・ハイリスク除外）
    elems += [
        _p("■ 注目銘柄　詳細分析", s_sub),
        HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=6),
    ]

    detail_targets = [r for r in results
                      if r["グロース評価スコア"] >= 4 and not r.get("ハイリスク")][:12]
    # 推奨銘柄はPART1で既に詳細表示済みなので、重複を省略しない（情報が異なる）
    print(f"\n[PART2] 詳細分析生成中... ({len(detail_targets)}銘柄)")

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
        desc_en  = (info.get("longBusinessSummary") or "")[:300]
        currency = "円" if market == "JP" else "USD"

        # 決算日取得
        try:
            ed_df = yf.Ticker(ticker).earnings_dates
            if ed_df is not None and len(ed_df) > 0:
                idx_ed = ed_df.index
                idx_ed = idx_ed.tz_localize(None) if idx_ed.tz is None else idx_ed.tz_convert(None)
                earnings_set_d = set(pd.DatetimeIndex(idx_ed).normalize().tolist())
            else:
                earnings_set_d = set()
        except Exception:
            earnings_set_d = set()

        raw_df = stock_data.get(ticker)

        hd_style = _style(f"p2_hd{i}", fontSize=11, fontName=_FONT_BOLD,
                          textColor=C_NAVY, spaceAfter=2, leading=16)
        sm_style = _style(f"p2_sm{i}", fontSize=8, leading=12, spaceAfter=2,
                          textColor=colors.HexColor("#555555"))
        elems += [
            _p(f"【{i}】{ticker}　{name}", hd_style),
            _p(f"市場: {'国内株（東証）' if market=='JP' else '米国株（NYSE/NASDAQ）'}　"
               f"セクター: {sector}　業種: {industry}　"
               f"グロース評価スコア: {r['グロース評価スコア']}", sm_style),
            Spacer(1, 1*mm),
        ]

        # 事業概要（翻訳）
        if desc_en:
            desc_ja = _translate(desc_en, 300)
            if len(info.get("longBusinessSummary", "")) > 300:
                desc_ja += "…"
            elems += [
                _p("◆ 事業概要",
                   _style(f"p2_dh{i}", fontName=_FONT_BOLD, fontSize=8.5,
                          leading=12, textColor=C_BLUE, spaceAfter=1)),
                _p(desc_ja, _style(f"p2_desc{i}", fontSize=8, leading=12,
                                   spaceAfter=2, textColor=colors.HexColor("#333333"))),
                Spacer(1, 1*mm),
            ]

        # 財務指標
        fin_data2 = [
            ["財務指標", "数値", "財務指標", "数値"],
            ["時価総額", mktcap_s, "売上成長率（前年比）", f"{rev_v:.1f}%"],
            ["株価収益率（PER）", f"{per_v:.1f}倍" if per_v else "N/A",
             "粗利率", f"{gross_v:.1f}%"],
            ["株価純資産倍率（PBR）", f"{pbr_v:.2f}倍" if pbr_v else "N/A",
             "営業利益率", f"{op_v:.1f}%"],
            ["自己資本利益率（ROE）", f"{roe_v:.1f}%",
             "総資産利益率（ROA）", f"{roa_v:.1f}%"],
            ["配当利回り", f"{div_v:.2f}%",
             "負債資本比率（D/E）", f"{de_v:.1f}" if de_v else "N/A"],
        ]
        elems.append(_tbl(fin_data2, [50*mm, 35*mm, 50*mm, 45*mm]))
        elems.append(Spacer(1, 2*mm))

        # チャート（変動分析付き）
        _sec_raw2  = (info.get("sector") or "") if market == "US" else ""
        img_buf2, sig_moves2 = generate_chart_with_annotations(
            ticker, market, raw_df, earnings_set_d,
            _bm_wret.get(market), _sec_etf_wret.get(_sec_raw2))

        if img_buf2:
            from reportlab.platypus import Image as RLImage
            elems += [
                _p("◆ 株価チャートと主要変動要因分析",
                   _style(f"p2_ch{i}", fontName=_FONT_BOLD, fontSize=8.5,
                          leading=12, textColor=C_BLUE, spaceAfter=2)),
                RLImage(img_buf2, width=178*mm, height=88*mm),
                Spacer(1, 1*mm),
            ]
            if sig_moves2:
                elems += _build_move_blocks(sig_moves2)
            elems.append(Spacer(1, 2*mm))

        # 直近ニュース（翻訳）
        try:
            news_list_d = yf.Ticker(ticker).news or []
            news_shown  = []
            for n in news_list_d[:15]:
                title  = n.get("title") or (n.get("content") or {}).get("title", "")
                pub_ts = n.get("providerPublishTime") or 0
                pub_dt = datetime.fromtimestamp(pub_ts).strftime("%Y/%m/%d") if pub_ts else "－"
                if title:
                    news_shown.append([pub_dt, _translate(title, 100)])
                if len(news_shown) >= 5: break
            if news_shown:
                nd = [["直近ニュース"]] + [[f"[{d}]　{t}"] for d, t in news_shown]
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

    # ━━ 免責事項 ━━
    elems += [
        PageBreak(),
        Spacer(1, 6*mm),
        HRFlowable(width="100%", thickness=1, color=C_NAVY, spaceAfter=4),
        _p("【免責事項】",
           _style("s_disc_h", fontName=_FONT_BOLD, fontSize=9,
                  textColor=C_NAVY, spaceAfter=2, leading=14)),
        _p("本レポートは情報提供を目的とした自動生成レポートであり、投資勧誘を意図するものではありません。"
           "掲載されている情報はスクリーニングアルゴリズムによる機械的な分析結果であり、"
           "投資判断の唯一の根拠とすることは適切ではありません。"
           "投資判断は必ずご自身の責任において行ってください。"
           "株式投資には価格変動リスクが伴い、元本の損失が生じる可能性があります。",
           _style("s_disc", fontSize=8, leading=12, spaceAfter=2,
                  textColor=colors.HexColor("#666666"))),
    ]

    doc.build(elems)
    print(f"[保存] 統合レポート: {path}")

    # results フォルダを最新5件に制限（古いものから削除）
    _results_root = os.path.dirname(OUTPUT_DIR)
    _runs = sorted(
        [d for d in os.listdir(_results_root)
         if os.path.isdir(os.path.join(_results_root, d))],
    )  # フォルダ名が YYYY-MM-DD_HHMM 形式なので辞書順 = 時系列順
    for _old in _runs[:-5]:
        _old_path = os.path.join(_results_root, _old)
        shutil.rmtree(_old_path, ignore_errors=True)
        print(f"[削除] 古いresult: {_old_path}")


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
if __name__ == "__main__":
    _results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_all, stock_data_raw = run_all_screens()

    jp_n = sum(1 for r in results_all if r["市場"] == "JP")
    us_n = len(results_all) - jp_n
    print(f"\nヒット: {len(results_all)}銘柄  (JP:{jp_n} / US:{us_n})")
    for r in results_all[:5]:
        print(f"  {r['市場']} {r['銘柄コード']:8s}  スコア:{r['グロース評価スコア']:5.1f}"
              f"  売上成長:{r.get('売上成長(%)',0):5.0f}%  {r['マッチ戦略'][:60]}")

    save_csv(results_all)
    save_report_pdf(results_all, stock_data_raw)
