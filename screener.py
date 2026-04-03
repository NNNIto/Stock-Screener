"""
ハイリターン中期投資スクリーナー v2.0
7戦略 × アナリスト目標株価ベース期待リターン
対象: 国内株式（東証）/ 米国株式（NYSE・NASDAQ）
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

# ── OpenAI（初期化のみ・PDF出力では未使用） ────────────────
_OPENAI_CLIENT = None
try:
    import openai as _openai_mod
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    if _oai_key:
        _OPENAI_CLIENT = _openai_mod.OpenAI(api_key=_oai_key)
        print("[GPT] OpenAI クライアント初期化完了")
except Exception as _e:
    print(f"[GPT] OpenAI 初期化スキップ: {_e}")

# ── チャートライブラリ（ヘッドレス設定のみ・描画には使わない） ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ("/mnt/c/Windows/Fonts/YuGothR.ttc",       "YuGothic",      "YuGothic"),
    ("/mnt/c/Windows/Fonts/YuGothB.ttc",       "YuGothicBold",  "YuGothicBold"),
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
C_GOLD   = colors.HexColor("#FFD700")
C_SILVER = colors.HexColor("#C0C0C0")
C_BRONZE = colors.HexColor("#CD7F32")
C_LGREEN = colors.HexColor("#dcfce7")
C_LORANGE = colors.HexColor("#fff3e0")

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
TODAY        = datetime.today().strftime("%Y-%m-%d")
RUN_DATETIME = datetime.today().strftime("%Y-%m-%d_%H%M")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "results", RUN_DATETIME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 市場別パラメータ（強化版）
# ─────────────────────────────────────────
MARKET_PARAMS = {
    "JP": {
        "mktcap_min":  30_000_000_000,  # 300億円
        "adv_min":     200_000_000,      # 2億円/日
        "price_min":   500,              # 500円
        "sma_s": "SMA25", "sma_m": "SMA75", "sma_l": "SMA200",
        "rev_grw_min": 0.15,
        "roe_min":     0.15,
        "gross_min":   0.35,
        "margin_min":  0.08,
    },
    "US": {
        "mktcap_min":  1_000_000_000,   # 10億USD
        "adv_min":     5_000_000,        # 500万USD/日
        "price_min":   15,               # 15 USD
        "sma_s": "SMA20", "sma_m": "SMA50", "sma_l": "SMA200",
        "rev_grw_min": 0.25,
        "roe_min":     0.20,
        "gross_min":   0.50,
        "margin_min":  0.10,
    },
}

# ─────────────────────────────────────────
# ダイナミックユニバース
# ─────────────────────────────────────────
FALLBACK_US = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","CRM","ADBE",
    "NOW","SNOW","DDOG","ZS","CRWD","NET","MDB","PLTR","ARM","ANET","FTNT",
    "PANW","HUBS","UBER","ABNB","AMD","QCOM","MU","MRVL","KLAC","LRCX","AMAT",
    "MPWR","SHOP","MELI","COIN","SOFI","LLY","MRNA","REGN","VRTX","ISRG",
    "COST","WMT","LULU","CAVA","AXON","TTD","CELH","IOT","GTLB","BILL","DUOL",
    "APP","TOST","KVYO","HIMS","RDDT","HOOD","SMCI","VST","NRG","CEG",
    "BABA","JD","PDD","ASML","SAP","MELI","SE","GRAB",
]

JP_UNIVERSE = [
    "6758.T","8035.T","6861.T","9984.T","4063.T","6954.T","7203.T","6902.T",
    "4543.T","6857.T","6920.T","4385.T","4689.T","6098.T","4484.T","9270.T",
    "4478.T","4433.T","3923.T","4448.T","4371.T","3769.T","6532.T","3064.T",
    "9983.T","3048.T","3092.T","4307.T","6976.T","3436.T","6326.T","6971.T",
    "7974.T","4502.T","4519.T","4568.T","4578.T","4523.T","4704.T","4661.T",
    "4751.T","4755.T","6501.T","6594.T","6762.T","6988.T","7716.T","7733.T",
    "7741.T","7751.T","8306.T","8309.T","8316.T","8411.T","8766.T","8802.T",
    "8830.T","9020.T","9022.T","9432.T","9433.T","9843.T","5713.T","8031.T",
    "8001.T","8058.T","2802.T","5401.T","8267.T","4901.T","3197.T","3086.T",
    "2782.T","7453.T","4752.T","6301.T","3659.T","4452.T","6146.T",
]


def _try_screener(query_name: str) -> list:
    """yfinance Screener APIを試行（複数アプローチ）"""
    # Try yf.Screener class first
    try:
        sc = yf.Screener()
        sc.set_predefined_body(query_name)
        resp = sc.response
        if isinstance(resp, dict) and "quotes" in resp:
            return [q.get("symbol", "") for q in resp["quotes"] if q.get("symbol")]
    except Exception:
        pass
    # Try yf.screen() function
    try:
        resp = yf.screen(query_name)
        if isinstance(resp, dict) and "quotes" in resp:
            return [q.get("symbol", "") for q in resp["quotes"] if q.get("symbol")]
    except Exception:
        pass
    return []


def fetch_dynamic_universe() -> tuple:
    """
    動的ユニバース取得。(tickers_list, market_map) を返す。
    """
    print("ユニバース取得中...")
    us_symbols = []
    queries = [
        "growth_technology_stocks",
        "undervalued_growth_stocks",
        "aggressive_small_caps",
        "most_actives",
    ]
    for q in queries:
        syms = _try_screener(q)
        # ドット無し・5文字以内のUSシンボルのみ
        syms = [s for s in syms if "." not in s and len(s) <= 5]
        us_symbols.extend(syms)
        time.sleep(0.3)

    us_symbols = list(dict.fromkeys(us_symbols))  # deduplicate

    if len(us_symbols) < 30:
        print(f"  スクリーナーから {len(us_symbols)} 銘柄 → フォールバックリスト使用")
        us_symbols = FALLBACK_US
    else:
        print(f"  スクリーナーから {len(us_symbols)} 銘柄取得")
        # fallbackも追加して網羅性を高める
        combined = list(dict.fromkeys(us_symbols + FALLBACK_US))
        us_symbols = combined

    all_tickers = list(dict.fromkeys(JP_UNIVERSE + us_symbols))
    market_map = {t: "JP" if t.endswith(".T") else "US" for t in all_tickers}

    n_jp = sum(1 for t in all_tickers if t.endswith(".T"))
    n_us = len(all_tickers) - n_jp
    print(f"対象銘柄数: {len(all_tickers)}  (国内: {n_jp}銘柄 / 米国: {n_us}銘柄)")
    return all_tickers, market_map


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
# マーケット環境取得（表紙用）
# ─────────────────────────────────────────
def fetch_market_env() -> dict:
    env = {}
    try:
        spx = yf.download("^GSPC", period="1y", progress=False, auto_adjust=True)
        if len(spx) >= 200:
            spx.columns = [c[0] if isinstance(c, tuple) else c for c in spx.columns]
            close = spx["Close"]
            sma200 = close.rolling(200).mean().iloc[-1]
            current = close.iloc[-1]
            env["SP500"] = round(float(current), 1)
            env["SP500_vs_SMA200"] = round((float(current) / float(sma200) - 1) * 100, 1)
    except Exception:
        pass
    try:
        vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        if len(vix) > 0:
            vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
            env["VIX"] = round(float(vix["Close"].iloc[-1]), 1)
    except Exception:
        pass
    try:
        nk = yf.download("^N225", period="5d", progress=False, auto_adjust=True)
        if len(nk) > 0:
            nk.columns = [c[0] if isinstance(c, tuple) else c for c in nk.columns]
            env["NIKKEI"] = round(float(nk["Close"].iloc[-1]), 0)
    except Exception:
        pass
    return env


# ─────────────────────────────────────────
# テクニカル指標計算
# ─────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df     = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    # SMA（JP用: 25/75/200、US用: 20/50/200）
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
# ベースフィルター
# ─────────────────────────────────────────
def passes_base_filter(df: pd.DataFrame, info: dict, market: str) -> bool:
    p = MARKET_PARAMS[market]
    r = get_latest(df)
    mktcap = info.get("marketCap", 0) or 0
    if mktcap > 0 and mktcap < p["mktcap_min"]:
        return False
    if r.Close < p["price_min"]:
        return False
    if not pd.isna(r.ADV20) and r.ADV20 < p["adv_min"]:
        return False
    return True


# ─────────────────────────────────────────
# ハイリスク判定
# ─────────────────────────────────────────
def is_high_risk(df: pd.DataFrame, info: dict) -> bool:
    try:
        latest = df.iloc[-1]
        rsi = float(latest.RSI14) if not pd.isna(latest.RSI14) else 0
        if rsi > 80:
            return True
        if not pd.isna(latest.DEV_SMA200) and latest.DEV_SMA200 > 60:
            return True
        vol = df["Close"].pct_change().iloc[-60:].std() * np.sqrt(252)
        if not np.isnan(vol) and vol > 0.85:
            return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────
# 7戦略スクリーニング関数
# ─────────────────────────────────────────

def S1_超高成長BK(df, info, market) -> bool:
    """S1: 52週高値ブレイクアウト + 出来高急増 + ADX + 売上成長"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W:
            return False
        vol_base = r.VOL_MA50 if not pd.isna(r.VOL_MA50) else r.VOL_MA20
        if pd.isna(vol_base) or r.Volume < vol_base * 2.5:
            return False
        if pd.isna(r.ADX) or r.ADX < 30:
            return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200:
            return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"]:
            return False
        return True
    except Exception:
        return False


def S2_グロースクオリティ(df, info, market) -> bool:
    """S2: 売上成長 + ROE + 粗利率 + 完全MA順列 + RSI + ADX"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"]:
            return False
        if (info.get("returnOnEquity") or 0) < p["roe_min"]:
            return False
        if (info.get("grossMargins") or 0) < p["gross_min"]:
            return False
        s = r[p["sma_s"]]
        m = r[p["sma_m"]]
        l = r[p["sma_l"]]
        if any(pd.isna([s, m, l])):
            return False
        if not (r.Close > s > m > l):
            return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 72):
            return False
        if pd.isna(r.ADX) or r.ADX < 25:
            return False
        return True
    except Exception:
        return False


def S3_CANSLIM強化(df, info, market) -> bool:
    """S3: CANSLIM強化版（ROE + 新高値 + 出来高 + 60日リターン）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        roe_min = 0.17 if market == "US" else 0.15
        if (info.get("returnOnEquity") or 0) < roe_min:
            return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"] * 0.8:
            return False
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W:
            return False
        vol_base = r.VOL_MA50 if not pd.isna(r.VOL_MA50) else r.VOL_MA20
        if pd.isna(vol_base) or r.Volume < vol_base * 2.0:
            return False
        if pd.isna(r.RET_60D) or r.RET_60D < 15:
            return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200:
            return False
        return True
    except Exception:
        return False


def S4_Rule50(df, info, market) -> bool:
    """S4: Rule of 50 (成長率+営業利益率 >= 50/40) + 粗利率 + MA + RSI"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        rev_grw = (info.get("revenueGrowth") or 0) * 100
        op_margin = (info.get("operatingMargins") or 0) * 100
        gross = (info.get("grossMargins") or 0)
        rule_threshold = 50 if market == "US" else 40
        if (rev_grw + op_margin) < rule_threshold:
            return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"]:
            return False
        gross_threshold = 0.55 if market == "US" else 0.45
        if gross < gross_threshold:
            return False
        sma_m = r[p["sma_m"]]
        if pd.isna(sma_m) or r.Close <= sma_m:
            return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 72):
            return False
        return True
    except Exception:
        return False


def S5_PEG割安高成長(df, info, market) -> bool:
    """S5: PEG割安 + PER適正 + 売上成長 + MA + RSI + ADX"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        peg = info.get("pegRatio")
        if peg is None:
            return False
        peg_max = 1.0 if market == "US" else 1.5
        if not (0 < peg <= peg_max):
            return False
        per = info.get("trailingPE") or 0
        if per < 5:
            return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"] * 0.8:
            return False
        sma_m = r[p["sma_m"]]
        if pd.isna(sma_m) or r.Close <= sma_m:
            return False
        if pd.isna(r.RSI14) or r.RSI14 < 45:
            return False
        if pd.isna(r.ADX) or r.ADX < 20:
            return False
        return True
    except Exception:
        return False


def S6_モメンタム最強(df, info, market) -> bool:
    """S6: 60日/20日リターン + ADX + SMA200上 + 出来高増加"""
    try:
        r = get_latest(df)
        if pd.isna(r.RET_60D) or r.RET_60D < 20:
            return False
        if pd.isna(r.RET_20D) or r.RET_20D < 6:
            return False
        if pd.isna(r.ADX) or r.ADX < 28:
            return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200:
            return False
        vol_base = r.VOL_MA20 if not pd.isna(r.VOL_MA20) else None
        if vol_base is None or r.Volume < vol_base * 1.3:
            return False
        return True
    except Exception:
        return False


def S7_アナリスト上値余地(df, info, market) -> bool:
    """S7: アナリスト目標株価上値余地 + アナリスト数 + 売上成長 + MA + RSI"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        target_mean = info.get("targetMeanPrice")
        if not target_mean or target_mean <= 0:
            return False
        current = float(r.Close)
        if current <= 0:
            return False
        upside = (target_mean - current) / current
        upside_min = 0.25 if market == "US" else 0.20
        if upside < upside_min:
            return False
        if (info.get("numberOfAnalystOpinions") or 0) < 3:
            return False
        if (info.get("revenueGrowth") or 0) < p["rev_grw_min"] * 0.6:
            return False
        sma_m = r[p["sma_m"]]
        if pd.isna(sma_m) or r.Close <= sma_m:
            return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 75):
            return False
        return True
    except Exception:
        return False


STRATEGIES = {
    "S1_超高成長BK":        S1_超高成長BK,
    "S2_グロースクオリティ": S2_グロースクオリティ,
    "S3_CANSLIM強化":        S3_CANSLIM強化,
    "S4_Rule50":             S4_Rule50,
    "S5_PEG割安高成長":      S5_PEG割安高成長,
    "S6_モメンタム最強":     S6_モメンタム最強,
    "S7_アナリスト上値余地": S7_アナリスト上値余地,
}


# ─────────────────────────────────────────
# アナリスト期待リターン計算
# ─────────────────────────────────────────
def calc_expected_returns(target_mean, current_price):
    """(ret_3m, ret_6m, ret_1y) を返す。データ不足時は (None, None, None)"""
    if not target_mean or target_mean <= 0 or current_price <= 0:
        return None, None, None
    ret_1y = (target_mean - current_price) / current_price * 100
    ret_6m = ret_1y * 0.60
    ret_3m = ret_1y * 0.35
    return round(ret_3m, 1), round(ret_6m, 1), round(ret_1y, 1)



# ─────────────────────────────────────────
# セクター日本語変換
# ─────────────────────────────────────────
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
    return _SECTOR_JP.get(name or "", name or "N/A")


# ─────────────────────────────────────────
# 翻訳
# ─────────────────────────────────────────
_trans_cache: dict = {}


def _translate(text: str, max_len: int = 400) -> str:
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
# 通貨フォーマット
# ─────────────────────────────────────────
def fmt_price(value, market: str) -> str:
    if value is None:
        return "N/A"
    if market == "JP":
        return f"¥{value:,.0f}"
    else:
        if value >= 1000:
            return f"${value:,.0f}"
        else:
            return f"${value:,.2f}"


def fmt_mktcap(value, market: str) -> str:
    if not value or value <= 0:
        return "N/A"
    if market == "JP":
        if value >= 1e12:
            return f"¥{value/1e12:.1f}兆"
        else:
            return f"¥{value/1e8:.0f}億"
    else:
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.1f}B"
        else:
            return f"${value/1e6:.0f}M"


def fmt_rec_key(rec_key: str) -> str:
    mapping = {
        "strong_buy":  "強い買い",
        "buy":         "買い",
        "hold":        "中立",
        "sell":        "売り",
        "strong_sell": "強い売り",
        "":            "N/A",
    }
    return mapping.get(rec_key or "", rec_key or "N/A")


def ret_color(ret):
    if ret is None:
        return C_BLACK
    if ret >= 30:
        return C_GREEN
    elif ret >= 10:
        return C_BLUE
    elif ret < 0:
        return C_RED
    return C_BLACK


# ─────────────────────────────────────────
# PDF ヘルパー関数
# ─────────────────────────────────────────

# テーブルセル用 ParagraphStyle
_cs_hdr     = ParagraphStyle("_csh",   fontName=_FONT_BOLD,   fontSize=8.5,
                              leading=12, textColor=C_WHITE,  wordWrap="CJK")
_cs_body    = ParagraphStyle("_csb",   fontName=_FONT_NORMAL, fontSize=8.5,
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


def _safe(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _p(text, s):
    return Paragraph(_safe(text), s)


def _tbl(data, col_widths, extra_styles=None, hdr_bg=C_NAVY, subhdr_rows=None):
    """
    テキスト折り返し対応テーブル。
    セルは Paragraph に変換して wordWrap='CJK' を保証する。
    """
    processed = []
    for ri, row in enumerate(data):
        new_row = []
        for cell in row:
            if isinstance(cell, Paragraph):
                new_row.append(cell)
            else:
                st = _cs_hdr if ri == 0 else _cs_body
                new_row.append(Paragraph(_safe(str(cell)), st))
        processed.append(new_row)

    t = Table(processed, colWidths=col_widths, repeatRows=1)
    base = [
        ("BACKGROUND",     (0, 0),  (-1, 0),  hdr_bg),
        ("ROWBACKGROUNDS", (0, 1),  (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",           (0, 0),  (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",         (0, 0),  (-1, -1), "TOP"),
        ("TOPPADDING",     (0, 0),  (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0),  (-1, -1), 4),
        ("LEFTPADDING",    (0, 0),  (-1, -1), 5),
        ("RIGHTPADDING",   (0, 0),  (-1, -1), 5),
    ]
    if subhdr_rows:
        for row_i, bg in subhdr_rows:
            base += [("BACKGROUND", (0, row_i), (-1, row_i), bg)]
            for ci in range(len(processed[row_i])):
                if not isinstance(data[row_i][ci], Paragraph):
                    processed[row_i][ci] = Paragraph(
                        _safe(str(data[row_i][ci])), _cs_lblue)
    if extra_styles:
        base += extra_styles
    t.setStyle(TableStyle(base))
    return t


def _color_cell(text: str, color) -> Paragraph:
    """指定色のテーブルセル用 Paragraph"""
    st = ParagraphStyle("_cc", fontName=_FONT_NORMAL, fontSize=8.5,
                        leading=12, textColor=color, wordWrap="CJK")
    return Paragraph(_safe(text), st)


def _bold_cell(text: str, color=C_BLACK, size=8.5) -> Paragraph:
    st = ParagraphStyle("_bc", fontName=_FONT_BOLD, fontSize=size,
                        leading=12, textColor=color, wordWrap="CJK")
    return Paragraph(_safe(text), st)


def _section_header(text: str, bg=C_NAVY, fg=C_WHITE, font_size=10) -> Table:
    """帯状セクションヘッダー"""
    st = ParagraphStyle("_sh", fontName=_FONT_BOLD, fontSize=font_size,
                        leading=15, textColor=fg, wordWrap="CJK")
    t = Table([[Paragraph(_safe(text), st)]], colWidths=[180 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t


# ─────────────────────────────────────────
# メインスクリーニング実行
# ─────────────────────────────────────────
def run_all_screens(tickers: list, market_map: dict):
    """スクリーニング実行。results リストを返す"""
    print("\nスクリーニング実行中...")
    stock_data_raw = fetch_stock_data(tickers, period="2y")

    results = []
    total = len(stock_data_raw)
    for idx, (ticker, raw_df) in enumerate(stock_data_raw.items(), 1):
        try:
            market = market_map.get(ticker, "US")
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

            if not hit:
                continue

            r = df.iloc[-1]
            p = MARKET_PARAMS[market]

            # 基本情報
            mktcap   = info.get("marketCap") or 0
            close_v  = float(r.Close)
            high_52w = float(r.HIGH_52W) if not pd.isna(r.HIGH_52W) else None
            low_52w  = float(r.LOW_52W)  if not pd.isna(r.LOW_52W)  else None
            sma_m_v  = float(r[p["sma_m"]]) if not pd.isna(r[p["sma_m"]]) else None
            sma_l_v  = float(r[p["sma_l"]]) if not pd.isna(r[p["sma_l"]]) else None

            # SMA状況
            if sma_m_v and sma_l_v and close_v > sma_m_v > sma_l_v:
                sma_status = "上昇順列"
            elif sma_l_v and close_v > sma_l_v:
                sma_status = "SMA上"
            else:
                sma_status = "SMA下"

            # アナリストデータ
            target_mean  = info.get("targetMeanPrice")
            target_high  = info.get("targetHighPrice")
            target_low   = info.get("targetLowPrice")
            n_analysts   = info.get("numberOfAnalystOpinions") or 0
            rec_key      = info.get("recommendationKey") or ""

            ret_3m, ret_6m, ret_1y = calc_expected_returns(target_mean, close_v)

            # アナリスト推奨内訳
            buy_count = hold_count = sell_count = 0
            try:
                rs = yf.Ticker(ticker).recommendations_summary
                if rs is not None and not rs.empty:
                    row_rs = rs.iloc[0]
                    buy_count  = int(row_rs.get("strongBuy", 0)) + int(row_rs.get("buy", 0))
                    hold_count = int(row_rs.get("hold", 0))
                    sell_count = int(row_rs.get("sell", 0)) + int(row_rs.get("strongSell", 0))
            except Exception:
                buy_count = hold_count = sell_count = 0

            # FCFマージン
            fcf          = info.get("freeCashflow")
            total_rev    = info.get("totalRevenue")
            fcf_margin   = None
            if fcf and total_rev and total_rev > 0:
                fcf_margin = round((fcf / total_rev) * 100, 1)

            rec = {
                "市場":             market,
                "銘柄コード":       ticker,
                "銘柄名":           info.get("longName") or info.get("shortName") or ticker,
                "セクター":         info.get("sector") or "N/A",
                "業種":             info.get("industry") or "N/A",
                "現在値":           close_v,
                "時価総額":         mktcap,
                "52W高値":          high_52w,
                "52W安値":          low_52w,
                "高値比(%)":        round((close_v / high_52w - 1) * 100, 1) if high_52w else None,
                # アナリスト
                "目標株価_平均":    target_mean,
                "目標株価_最高":    target_high,
                "目標株価_最低":    target_low,
                "アナリスト数":     n_analysts,
                "推奨_買い":        buy_count,
                "推奨_中立":        hold_count,
                "推奨_売り":        sell_count,
                "推奨区分":         rec_key,
                # 期待リターン
                "期待リターン_3M(%)": ret_3m,
                "期待リターン_6M(%)": ret_6m,
                "期待リターン_1Y(%)": ret_1y,
                # ファンダメンタルズ
                "売上成長(%)":      round((info.get("revenueGrowth")    or 0) * 100, 1),
                "ROE(%)":           round((info.get("returnOnEquity")   or 0) * 100, 1),
                "粗利率(%)":        round((info.get("grossMargins")     or 0) * 100, 1),
                "営業利益率(%)":    round((info.get("operatingMargins") or 0) * 100, 1),
                "PER":              round(info.get("trailingPE") or 0, 1) or None,
                "PEG":              round(info.get("pegRatio")   or 0, 2) or None,
                "PSR":              round(info.get("priceToSalesTrailing12Months") or 0, 2) or None,
                "DE比率":           round(info.get("debtToEquity") or 0, 2) or None,
                "FCFマージン(%)":   fcf_margin,
                # テクニカル
                "RSI14":            round(float(r.RSI14), 1) if not pd.isna(r.RSI14) else None,
                "ADX":              round(float(r.ADX),   1) if not pd.isna(r.ADX)   else None,
                "SMA状況":          sma_status,
                "60日リターン(%)":  round(float(r.RET_60D), 1) if not pd.isna(r.RET_60D) else None,
                "20日リターン(%)":  round(float(r.RET_20D), 1) if not pd.isna(r.RET_20D) else None,
                "ハイリスク":       is_high_risk(df, info),
                # 戦略
                "マッチ戦略数":     len(hit),
                "マッチ戦略":       " | ".join(hit),
                "_info":            info,
            }
            results.append(rec)

        except Exception as e:
            pass

    results_sorted = sorted(results,
        key=lambda x: (-(x.get("期待リターン_1Y(%)") or -999), -x["マッチ戦略数"]))
    return results_sorted, stock_data_raw


# ─────────────────────────────────────────
# CSV保存
# ─────────────────────────────────────────
def save_csv(results: list):
    if not results:
        return

    rows = [{
        "市場":               r["市場"],
        "銘柄コード":         r["銘柄コード"],
        "銘柄名":             r.get("銘柄名", ""),
        "現在値":             r["現在値"],
        "目標株価_平均":      r.get("目標株価_平均", ""),
        "期待リターン_3M(%)": r.get("期待リターン_3M(%)", ""),
        "期待リターン_6M(%)": r.get("期待リターン_6M(%)", ""),
        "期待リターン_1Y(%)": r.get("期待リターン_1Y(%)", ""),
        "推奨区分":           r.get("推奨区分", ""),
        "RSI14":              r.get("RSI14", ""),
        "ADX":                r.get("ADX", ""),
        "売上成長(%)":        r.get("売上成長(%)", ""),
        "粗利率(%)":          r.get("粗利率(%)", ""),
        "ROE(%)":             r.get("ROE(%)", ""),
        "PER":                r.get("PER", ""),
        "PEG":                r.get("PEG", ""),
        "PSR":                r.get("PSR", ""),
        "FCFマージン(%)":     r.get("FCFマージン(%)", ""),
        "DE比率":             r.get("DE比率", ""),
        "マッチ戦略数":       r["マッチ戦略数"],
        "マッチ戦略":         r["マッチ戦略"],
        "ハイリスク":         r.get("ハイリスク", False),
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
            "戦略":       sname,
            "国内ヒット数": len(jp_s),
            "米国ヒット数": len(us_s),
            "合計":       len(hits),
            "国内銘柄":   "、".join(jp_s),
            "米国銘柄":   "、".join(us_s),
        })
    pd.DataFrame(strat_rows).to_csv(
        os.path.join(OUTPUT_DIR, "screening_summary.csv"),
        index=False, encoding="utf-8-sig")

    print(f"[保存] CSV: {OUTPUT_DIR}")


# ─────────────────────────────────────────
# PDF生成: ランキングページ用ヘルパー
# ─────────────────────────────────────────
def _build_ranking_page(elems, results: list, s_sub, s_small):
    """期待リターン統合ランキングページ（1Y降順・全銘柄・3M/6M/1Yカラム）"""
    elems.append(_section_header(
        "期待リターン ランキング（アナリスト平均目標株価ベース）", bg=C_NAVY))
    elems.append(Spacer(1, 2 * mm))
    elems.append(_p(
        "※ アナリスト平均目標株価から1年後期待リターンを算出。3ヶ月後=×35%、6ヶ月後=×60%で推定。"
        "投資判断の参考にとどめてください。", s_small))
    elems.append(Spacer(1, 3 * mm))

    sorted_r = sorted(
        results,
        key=lambda x: (-(x.get("期待リターン_1Y(%)") or -999), -x["マッチ戦略数"])
    )

    if not sorted_r:
        elems.append(_p("ヒット銘柄がありません", s_small))
        elems.append(PageBreak())
        return

    hdr = ["順位", "ティッカー", "企業名", "セクター", "現在値", "目標株価",
           "3ヶ月後", "6ヶ月後", "1年後", "戦略数"]
    col_w = [10*mm, 16*mm, 38*mm, 26*mm, 18*mm, 18*mm, 18*mm, 18*mm, 18*mm, 12*mm]
    data = [hdr]

    rank_bg = {1: C_GOLD, 2: C_SILVER, 3: C_BRONZE}
    extra_styles = []

    for rank, r in enumerate(sorted_r, 1):
        market = r["市場"]
        r3  = r.get("期待リターン_3M(%)")
        r6  = r.get("期待リターン_6M(%)")
        r1y = r.get("期待リターン_1Y(%)")

        def fmt_ret(v):
            return f"{v:+.1f}%" if v is not None else "N/A"

        name_s   = (r.get("銘柄名") or r["銘柄コード"])[:18]
        sector_s = _jp_sector(r.get("セクター") or "")[:14]

        data.append([
            f"#{rank}",
            r["銘柄コード"],
            name_s,
            sector_s,
            fmt_price(r.get("現在値"), market),
            fmt_price(r.get("目標株価_平均"), market),
            fmt_ret(r3),
            fmt_ret(r6),
            fmt_ret(r1y),
            f"{r['マッチ戦略数']}/7",
        ])

        row_idx = rank
        if rank in rank_bg:
            extra_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx), rank_bg[rank]))
        # 期待リターン列3本に色付け (col 6,7,8)
        for col_i, val in [(6, r3), (7, r6), (8, r1y)]:
            c = ret_color(val)
            if c != C_BLACK:
                extra_styles.append(("TEXTCOLOR", (col_i, row_idx), (col_i, row_idx), c))

    elems.append(_tbl(data, col_w, extra_styles=extra_styles))
    elems.append(PageBreak())


# ─────────────────────────────────────────
# PDF生成: 個別銘柄詳細ページ
# ─────────────────────────────────────────
def _build_stock_detail(elems, r: dict, s_small):
    """個別銘柄詳細ページを構築"""
    market   = r["市場"]
    ticker   = r["銘柄コード"]
    name     = r.get("銘柄名") or ticker
    sector   = _jp_sector(r.get("セクター") or "")
    industry = _jp_sector(r.get("業種") or "")
    info     = r.get("_info") or {}

    # Section 1: ヘッダー帯
    hdr_text = (f"{ticker}  {name}  |  {sector} / {industry}"
                f"  |  {'国内株（東証）' if market == 'JP' else '米国株'}")
    elems.append(_section_header(hdr_text, bg=C_NAVY, fg=C_WHITE, font_size=9))
    elems.append(Spacer(1, 2 * mm))

    # Section 2: アナリスト予測 + 現在株価情報 (2列)
    target_mean = r.get("目標株価_平均")
    target_high = r.get("目標株価_最高")
    target_low  = r.get("目標株価_最低")
    n_analysts  = r.get("アナリスト数") or 0
    buy_c       = r.get("推奨_買い") or 0
    hold_c      = r.get("推奨_中立") or 0
    sell_c      = r.get("推奨_売り") or 0
    rec_key_jp  = fmt_rec_key(r.get("推奨区分") or "")

    close_v     = r.get("現在値")
    mktcap      = r.get("時価総額")
    high_52w    = r.get("52W高値")
    low_52w     = r.get("52W安値")
    ratio_52w   = r.get("高値比(%)")

    ratio_color = C_RED if (ratio_52w is not None and ratio_52w < -20) else C_BLACK
    ratio_str   = f"{ratio_52w:+.1f}%" if ratio_52w is not None else "N/A"

    _cs_lbl = ParagraphStyle("_sec2lbl", fontName=_FONT_BOLD,   fontSize=8,
                              leading=12, textColor=C_NAVY,  wordWrap="CJK")
    _cs_val = ParagraphStyle("_sec2val", fontName=_FONT_NORMAL, fontSize=8.5,
                              leading=12, textColor=C_BLACK, wordWrap="CJK")

    def lbl(t): return Paragraph(_safe(t), _cs_lbl)
    def val(t, color=C_BLACK):
        st = ParagraphStyle("_v", fontName=_FONT_NORMAL, fontSize=8.5,
                            leading=12, textColor=color, wordWrap="CJK")
        return Paragraph(_safe(t), st)

    analyst_rows = [
        [lbl("平均目標株価"),  val(fmt_price(target_mean, market))],
        [lbl("最高目標株価"),  val(fmt_price(target_high, market))],
        [lbl("最低目標株価"),  val(fmt_price(target_low,  market))],
        [lbl("推奨"),          val(rec_key_jp)],
    ]
    price_rows = [
        [lbl("現在値"),       val(fmt_price(close_v, market))],
        [lbl("時価総額"),     val(fmt_mktcap(mktcap, market))],
        [lbl("52週高値"),     val(fmt_price(high_52w, market))],
        [lbl("52週安値"),     val(fmt_price(low_52w, market))],
        [lbl("高値比"),       val(ratio_str, ratio_color)],
    ]

    # 2列並びに整形（左: アナリスト、右: 株価情報）
    two_col_data = []
    max_rows = max(len(analyst_rows), len(price_rows))
    for i in range(max_rows):
        al = analyst_rows[i] if i < len(analyst_rows) else [lbl(""), val("")]
        pr = price_rows[i]   if i < len(price_rows)   else [lbl(""), val("")]
        two_col_data.append([al[0], al[1], pr[0], pr[1]])

    two_col_t = Table(two_col_data, colWidths=[28*mm, 62*mm, 28*mm, 62*mm])
    two_col_t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",     (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 4),
    ]))
    elems.append(two_col_t)
    elems.append(Spacer(1, 2 * mm))

    # Section 3: 期待リターン帯
    ret_3m = r.get("期待リターン_3M(%)")
    ret_6m = r.get("期待リターン_6M(%)")
    ret_1y = r.get("期待リターン_1Y(%)")

    if target_mean:
        def pret(v):
            return f"{v:+.1f}%" if v is not None else "N/A"
        ret_text = (f"期待リターン（アナリスト目標株価ベース推定）"
                    f"　3ヶ月後: {pret(ret_3m)}"
                    f"　6ヶ月後: {pret(ret_6m)}"
                    f"　1年後: {pret(ret_1y)}")
        elems.append(_section_header(ret_text, bg=C_LBLUE, fg=C_NAVY, font_size=9))
    else:
        elems.append(_section_header("アナリスト目標株価データなし", bg=C_LBLUE, fg=C_NAVY, font_size=9))
    elems.append(Spacer(1, 2 * mm))

    # Section 4: ファンダメンタルズ（3列 × 3行）
    def fund_val(v, good_thresh, bad_thresh=None, suffix="", fmt="{:.1f}"):
        if v is None:
            return val("N/A")
        s = fmt.format(v) + suffix
        if bad_thresh is not None and v < bad_thresh:
            return val(s, C_RED)
        if v >= good_thresh:
            return val(s, C_GREEN)
        return val(s)

    rev   = r.get("売上成長(%)")
    roe   = r.get("ROE(%)")
    gross = r.get("粗利率(%)")
    op    = r.get("営業利益率(%)")
    fcf   = r.get("FCFマージン(%)")
    de    = r.get("DE比率")
    per   = r.get("PER")
    peg   = r.get("PEG")
    psr   = r.get("PSR")

    _cs_flbl = ParagraphStyle("_flbl", fontName=_FONT_BOLD,   fontSize=7.5,
                               leading=11, textColor=C_NAVY,  wordWrap="CJK")

    def flbl(t): return Paragraph(_safe(t), _cs_flbl)

    fund_data = [
        [flbl("売上成長"), fund_val(rev,   30, None, "%"),
         flbl("ROE"),      fund_val(roe,   20, None, "%"),
         flbl("粗利率"),   fund_val(gross, 50, None, "%")],
        [flbl("営業利益率"), fund_val(op,  15, None, "%"),
         flbl("FCFマージン"), fund_val(fcf, 15, None, "%"),
         flbl("DE比率"),   fund_val(de,  100, 200, "", "{:.1f}") if de else val("N/A")],
        [flbl("PER"),      fund_val(per,  0,  None, "倍", "{:.0f}") if per else val("N/A"),
         flbl("PEG"),      fund_val(peg,  0,  None, "",   "{:.2f}") if peg else val("N/A"),
         flbl("PSR"),      fund_val(psr,  0,  None, "倍", "{:.1f}") if psr else val("N/A")],
    ]
    fund_t = Table(fund_data, colWidths=[18*mm, 42*mm, 18*mm, 42*mm, 18*mm, 42*mm])
    fund_t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 4),
    ]))
    elems.append(fund_t)
    elems.append(Spacer(1, 2 * mm))

    # Section 5: テクニカルサマリー（1行5セル）
    rsi_v = r.get("RSI14")
    adx_v = r.get("ADX")
    sma_s = r.get("SMA状況") or "N/A"
    r60   = r.get("60日リターン(%)")
    r20   = r.get("20日リターン(%)")

    rsi_str = f"RSI14: {rsi_v:.1f}" if rsi_v else "RSI14: N/A"
    adx_str = f"ADX: {adx_v:.1f}"   if adx_v else "ADX: N/A"
    r60_str = f"60日: {r60:+.1f}%"  if r60 is not None else "60日: N/A"
    r20_str = f"20日: {r20:+.1f}%"  if r20 is not None else "20日: N/A"

    rsi_color  = C_RED if (rsi_v and rsi_v > 70) else (C_GREEN if (rsi_v and 50 <= rsi_v <= 70) else C_BLACK)
    adx_color  = C_GREEN if (adx_v and adx_v >= 25) else C_BLACK
    sma_color  = C_GREEN if sma_s == "上昇順列" else (C_ORANGE if sma_s == "SMA上" else C_RED)
    r60_color  = ret_color(r60)
    r20_color  = ret_color(r20)

    tech_data = [[
        Paragraph(_safe(rsi_str), ParagraphStyle("_t1", fontName=_FONT_BOLD, fontSize=8.5, leading=12, textColor=rsi_color, wordWrap="CJK")),
        Paragraph(_safe(adx_str), ParagraphStyle("_t2", fontName=_FONT_BOLD, fontSize=8.5, leading=12, textColor=adx_color, wordWrap="CJK")),
        Paragraph(_safe(f"MA: {sma_s}"), ParagraphStyle("_t3", fontName=_FONT_BOLD, fontSize=8.5, leading=12, textColor=sma_color, wordWrap="CJK")),
        Paragraph(_safe(r60_str), ParagraphStyle("_t4", fontName=_FONT_BOLD, fontSize=8.5, leading=12, textColor=r60_color, wordWrap="CJK")),
        Paragraph(_safe(r20_str), ParagraphStyle("_t5", fontName=_FONT_BOLD, fontSize=8.5, leading=12, textColor=r20_color, wordWrap="CJK")),
    ]]
    tech_t = Table(tech_data, colWidths=[36*mm, 32*mm, 36*mm, 38*mm, 38*mm])
    tech_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LBLUE),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))
    elems.append(tech_t)
    elems.append(Spacer(1, 2 * mm))

    # Section 6: マッチ戦略
    n_hit      = r.get("マッチ戦略数") or 0
    strat_text = r.get("マッチ戦略") or ""
    strat_tags = strat_text.replace(" | ", "  |  ")
    elems.append(_p(f"マッチ戦略 ({n_hit}/7):  {strat_tags}",
                    _style(f"_strat_{ticker}", fontSize=8.5, fontName=_FONT_BOLD,
                           textColor=C_BLUE, spaceAfter=2, leading=13)))

    # Section 7: ハイリスク警告
    if r.get("ハイリスク"):
        warn_text = "ハイリスク警告: RSI過熱 / 高ボラティリティ / SMA200から大幅乖離等"
        elems.append(_section_header(warn_text, bg=C_LORANGE, fg=C_ORANGE, font_size=9))

    elems.append(Spacer(1, 3 * mm))
    elems.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#aaaaaa"), spaceAfter=4))


# ─────────────────────────────────────────
# PDF生成: メイン
# ─────────────────────────────────────────
def generate_pdf(results: list, market_env: dict,
                 n_screened_jp: int, n_screened_us: int):
    """完全新設計PDF生成（チャート・GPT分析なし）"""
    if not results:
        print("ヒット銘柄なし")
        return

    path = os.path.join(OUTPUT_DIR, "screening_report.pdf")
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)

    # 共通スタイル
    s_cover_title = _style("s_ct", fontSize=20, fontName=_FONT_BOLD,
                           textColor=C_NAVY, spaceAfter=4, leading=26)
    s_cover_sub   = _style("s_cs", fontSize=12, fontName=_FONT_BOLD,
                           textColor=C_BLUE, spaceAfter=3, leading=17)
    s_sub         = _style("s_sub", fontSize=12, fontName=_FONT_BOLD,
                           textColor=C_BLUE, spaceAfter=3, leading=17)
    s_body        = _style("s_body",  fontSize=9,  leading=14, spaceAfter=3)
    s_small       = _style("s_small", fontSize=8,  leading=12, spaceAfter=2,
                           textColor=colors.HexColor("#444444"))

    jp_hits = [r for r in results if r["市場"] == "JP"]
    us_hits = [r for r in results if r["市場"] == "US"]

    elems = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Page 1: 表紙
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elems += [
        Spacer(1, 12 * mm),
        _p("ハイリターン中期投資スクリーナー v2.0", s_cover_title),
        HRFlowable(width="100%", thickness=3, color=C_NAVY, spaceAfter=6),
        _p(f"実行日時: {datetime.today().strftime('%Y-%m-%d %H:%M')}　"
           f"対象: 国内株式（東証）/ 米国株式（NYSE・NASDAQ）", s_body),
        Spacer(1, 5 * mm),
    ]

    # ユニバース統計
    uni_data = [
        ["項目", "国内株（JP）", "米国株（US）", "合計"],
        ["スキャン銘柄数",
         f"{n_screened_jp}銘柄",
         f"{n_screened_us}銘柄",
         f"{n_screened_jp + n_screened_us}銘柄"],
        ["ヒット銘柄数",
         f"{len(jp_hits)}銘柄",
         f"{len(us_hits)}銘柄",
         f"{len(results)}銘柄"],
    ]
    elems.append(_tbl(uni_data, [70*mm, 35*mm, 35*mm, 40*mm]))
    elems.append(Spacer(1, 5 * mm))

    # マーケット環境
    env_rows = [["指数", "現在値", "状況"]]
    sp500 = market_env.get("SP500")
    sp_vs = market_env.get("SP500_vs_SMA200")
    if sp500:
        sp_status = f"200日SMA比 {sp_vs:+.1f}%" if sp_vs is not None else "N/A"
        env_rows.append(["S&P500", f"{sp500:,.1f}", sp_status])
    vix = market_env.get("VIX")
    if vix:
        vix_status = "恐怖指数高（警戒）" if vix >= 25 else "安定圏"
        env_rows.append(["VIX（恐怖指数）", f"{vix:.1f}", vix_status])
    nk = market_env.get("NIKKEI")
    if nk:
        env_rows.append(["日経225", f"{nk:,.0f}", "参考値"])

    if len(env_rows) > 1:
        elems.append(_p("■ マーケット環境", s_cover_sub))
        elems.append(_tbl(env_rows, [60*mm, 50*mm, 70*mm]))
        elems.append(Spacer(1, 5 * mm))

    # 戦略別ヒット数
    strat_data = [["戦略", "国内ヒット", "米国ヒット", "合計"]]
    for sname in STRATEGIES:
        hits = [r for r in results if sname in r["マッチ戦略"]]
        jp_s = sum(1 for r in hits if r["市場"] == "JP")
        us_s = sum(1 for r in hits if r["市場"] == "US")
        strat_data.append([sname, str(jp_s), str(us_s), str(len(hits))])

    elems.append(_p("■ 戦略別ヒット数", s_cover_sub))
    elems.append(_tbl(strat_data, [90*mm, 28*mm, 28*mm, 34*mm]))
    elems.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Page 2: 期待リターン 統合ランキング
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _build_ranking_page(elems, results, s_sub=s_sub, s_small=s_small)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Pages 5+: 個別銘柄詳細（1Y期待リターン降順）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    detail_results = sorted(results,
        key=lambda x: (-(x.get("期待リターン_1Y(%)") or -999), -x["マッチ戦略数"]))
    print(f"\n[PDF] 個別銘柄詳細生成中... ({len(detail_results)}銘柄)")

    for idx, r in enumerate(detail_results):
        print(f"  [{idx+1}/{len(detail_results)}] {r['銘柄コード']} ...")
        _build_stock_detail(elems, r, s_small)
        if idx < len(detail_results) - 1:
            elems.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 最終ページ: 免責事項
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elems += [
        PageBreak(),
        Spacer(1, 8 * mm),
        HRFlowable(width="100%", thickness=1.5, color=C_NAVY, spaceAfter=4),
        _p("【免責事項】",
           _style("s_disc_h", fontName=_FONT_BOLD, fontSize=10,
                  textColor=C_NAVY, spaceAfter=4, leading=15)),
        _p("本レポートは情報提供を目的とした自動生成レポートであり、投資勧誘を意図するものではありません。",
           _style("s_d1", fontSize=8.5, leading=13, spaceAfter=3,
                  textColor=colors.HexColor("#333333"))),
        _p("掲載されている情報はスクリーニングアルゴリズムによる機械的な分析結果であり、"
           "投資判断の唯一の根拠とすることは適切ではありません。",
           _style("s_d2", fontSize=8.5, leading=13, spaceAfter=3,
                  textColor=colors.HexColor("#333333"))),
        _p("アナリスト目標株価に基づく期待リターンは、アナリスト予測の単純線形推定であり、"
           "将来の株価を保証するものではありません。",
           _style("s_d3", fontSize=8.5, leading=13, spaceAfter=3,
                  textColor=colors.HexColor("#333333"))),
        _p("投資判断は必ずご自身の責任において行ってください。"
           "株式投資には価格変動リスクが伴い、元本の損失が生じる可能性があります。",
           _style("s_d4", fontSize=8.5, leading=13, spaceAfter=3,
                  textColor=colors.HexColor("#333333"))),
        Spacer(1, 4 * mm),
        _p(f"生成日時: {datetime.today().strftime('%Y-%m-%d %H:%M')}  |  "
           f"スクリーナー v2.0  |  7戦略 × アナリスト期待リターン",
           _style("s_d5", fontSize=7.5, leading=11, spaceAfter=2,
                  textColor=colors.HexColor("#888888"))),
    ]

    doc.build(elems)
    print(f"[保存] 統合レポート: {path}")

    # results フォルダを最新5件に制限
    _results_root = os.path.dirname(OUTPUT_DIR)
    _runs = sorted(
        [d for d in os.listdir(_results_root)
         if os.path.isdir(os.path.join(_results_root, d))]
    )
    for _old in _runs[:-5]:
        _old_path = os.path.join(_results_root, _old)
        shutil.rmtree(_old_path, ignore_errors=True)
        print(f"[削除] 古いresult: {_old_path}")


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # マーケット環境取得
    print("マーケット環境データ取得中...")
    market_env = fetch_market_env()
    sp500 = market_env.get("SP500")
    vix   = market_env.get("VIX")
    nk    = market_env.get("NIKKEI")
    print(f"  S&P500: {sp500}  VIX: {vix}  日経225: {nk}")

    # ダイナミックユニバース取得
    tickers, market_map = fetch_dynamic_universe()
    n_jp = sum(1 for t in tickers if t.endswith(".T"))
    n_us = len(tickers) - n_jp

    # スクリーニング実行
    results_all, stock_data_raw = run_all_screens(tickers, market_map)

    jp_n = sum(1 for r in results_all if r["市場"] == "JP")
    us_n = len(results_all) - jp_n
    print(f"\nヒット: {len(results_all)}銘柄  (JP:{jp_n} / US:{us_n})")
    for r in results_all[:5]:
        ret1y = r.get("期待リターン_1Y(%)")
        ret_s = f"{ret1y:+.1f}%" if ret1y is not None else "N/A"
        print(f"  {r['市場']} {r['銘柄コード']:8s}  1Y期待:{ret_s:>7s}"
              f"  戦略:{r['マッチ戦略数']}/7  {r['マッチ戦略'][:55]}")

    save_csv(results_all)
    generate_pdf(results_all, market_env, n_jp, n_us)

    # ── 実行後 Git コミット & プッシュ ──────────────────
    _repo = os.path.dirname(os.path.abspath(__file__))
    _dt   = datetime.today().strftime("%Y-%m-%d %H:%M")
    try:
        import subprocess as _sp
        _sp.run(["git", "-C", _repo, "add", "screener.py", "results/"],
                check=True)
        _sp.run(["git", "-C", _repo, "add", "-u"],
                check=True)
        _sp.run(["git", "-C", _repo, "commit", "-m",
                 f"スクリーニング結果: {_dt}"],
                check=True)
        _sp.run(["git", "-C", _repo, "push", "origin", "main"],
                check=True)
        print(f"[Git] プッシュ完了: スクリーニング結果: {_dt}")
    except Exception as _ge:
        print(f"[Git] プッシュ失敗（手動対応してください）: {_ge}")
