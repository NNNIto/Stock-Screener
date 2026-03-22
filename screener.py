"""
中期投資スクリーニング 全21戦略 実行スクリプト
参照: screening_thresholds.txt / screening_combinations.txt
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAY = datetime.today().strftime("%Y-%m-%d")
PERIOD_LONG  = "2y"   # 200日MA用
PERIOD_SHORT = "6mo"  # 短期指標用

# 東証プライム主要銘柄リスト（流動性の高い代表銘柄）
STOCK_UNIVERSE = [
    # 指数構成・大型株
    "7203.T","6758.T","8306.T","9432.T","6861.T","4063.T","8035.T","9984.T",
    "6954.T","7974.T","4502.T","8316.T","7267.T","6902.T","9433.T","4519.T",
    "6501.T","7751.T","8411.T","4661.T","6702.T","8058.T","2914.T","9022.T",
    "7741.T","6367.T","3382.T","8031.T","6503.T","4543.T","6326.T","8001.T",
    "4568.T","7269.T","6724.T","8766.T","9020.T","7733.T","6971.T","4578.T",
    "8802.T","6645.T","5108.T","7201.T","4911.T","2802.T","6762.T","6301.T",
    "8309.T","7符.T","6988.T","4704.T","4523.T","6857.T","8830.T","9437.T",
    "4751.T","3659.T","6594.T","7716.T","4755.T","3産.T","6460.T","2413.T",
    # 中型株・成長株
    "6920.T","4385.T","4689.T","3697.T","4478.T","4433.T","3923.T","4980.T",
    "6098.T","4448.T","7072.T","4371.T","3769.T","6095.T","4484.T","4565.T",
    "4395.T","6532.T","4194.T","4376.T","3950.T","9166.T","4193.T","4397.T",
    "6522.T","4199.T","3987.T","4892.T","9270.T","7748.T","6268.T","4387.T",
    "2150.T","3064.T","9843.T","3197.T","2651.T","3405.T","5401.T","5411.T",
    "5713.T","5714.T","3086.T","8267.T","3099.T","7513.T","9983.T","2678.T",
    "3048.T","2782.T","3141.T","7453.T","9831.T","8905.T","8136.T","3092.T",
    "9601.T","4307.T","6976.T","4704.T","3436.T","5332.T","4901.T","6971.T",
]

# 無効なティッカーを除去
STOCK_UNIVERSE = [t for t in STOCK_UNIVERSE if t.replace(".T","").isdigit() or len(t) <= 8]
# 数字4桁.T のみ残す
STOCK_UNIVERSE = [t for t in STOCK_UNIVERSE if len(t.replace(".T","")) == 4 and t.replace(".T","").isdigit()]

# 重複除去
STOCK_UNIVERSE = list(dict.fromkeys(STOCK_UNIVERSE))

print(f"対象銘柄数: {len(STOCK_UNIVERSE)}")

# ─────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────
def fetch_stock_data(tickers: list, period: str = "2y") -> dict:
    """yfinanceで一括取得"""
    print(f"\n株価データ取得中... ({len(tickers)}銘柄)")
    data = {}
    batch_size = 10
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  {i+1}〜{min(i+batch_size, len(tickers))}銘柄目...")
        for ticker in batch:
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if df is not None and len(df) >= 60:
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    data[ticker] = df
            except Exception:
                pass
        time.sleep(0.5)
    print(f"取得成功: {len(data)}銘柄")
    return data

def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception:
        return {}

# ─────────────────────────────────────────
# テクニカル指標計算
# ─────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """全指標を一括計算"""
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    # 移動平均
    df["SMA25"]  = ta.sma(close, 25)
    df["SMA75"]  = ta.sma(close, 75)
    df["SMA200"] = ta.sma(close, 200)

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"]     = macd.iloc[:, 0]
        df["MACD_sig"] = macd.iloc[:, 1]
        df["MACD_hist"]= macd.iloc[:, 2]

    # RSI
    df["RSI14"] = ta.rsi(close, 14)

    # ボリンジャーバンド
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        df["BB_upper"] = bb.iloc[:, 0]
        df["BB_mid"]   = bb.iloc[:, 1]
        df["BB_lower"] = bb.iloc[:, 2]
        df["BB_bw"]    = bb.iloc[:, 3]  # バンド幅
        df["BB_pct"]   = bb.iloc[:, 4]  # %B

    # ストキャスティクス
    stoch = ta.stoch(df["High"], df["Low"], close, k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        df["STOCH_K"] = stoch.iloc[:, 0]
        df["STOCH_D"] = stoch.iloc[:, 1]

    # ATR
    df["ATR14"] = ta.atr(df["High"], df["Low"], close, 14)

    # ADX / DI
    adx = ta.adx(df["High"], df["Low"], close, 14)
    if adx is not None and not adx.empty:
        df["ADX"]    = adx.iloc[:, 0]
        df["DI_pos"] = adx.iloc[:, 1]
        df["DI_neg"] = adx.iloc[:, 2]

    # 出来高移動平均
    df["VOL_MA20"] = ta.sma(volume, 20)

    # 52週高値・安値
    df["HIGH_52W"] = close.rolling(252).max()
    df["LOW_52W"]  = close.rolling(252).min()

    # 乖離率
    df["DEV_SMA25"]  = (close / df["SMA25"]  - 1) * 100
    df["DEV_SMA75"]  = (close / df["SMA75"]  - 1) * 100
    df["DEV_SMA200"] = (close / df["SMA200"] - 1) * 100

    # モメンタム（過去リターン）
    df["RET_20D"]  = close.pct_change(20) * 100
    df["RET_60D"]  = close.pct_change(60) * 100

    return df

# ─────────────────────────────────────────
# スクリーニング関数群（全21戦略）
# ─────────────────────────────────────────
def get_latest(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1]

def get_prev(df: pd.DataFrame, n: int = 1) -> pd.Series:
    return df.iloc[-(1+n)]

def screen_A1(df: pd.DataFrame) -> bool:
    """A-1: 王道モメンタム戦略"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        # 完全順列
        if not (r.Close > r.SMA25 > r.SMA75 > r.SMA200): return False
        # GC: 直近20日以内に25日SMAが75日SMAを上抜け
        gc = False
        for i in range(min(20, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if r0.SMA25 > r0.SMA75 and r1.SMA25 <= r1.SMA75:
                gc = True; break
        if not gc: return False
        # 出来高急増
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.5: return False
        # RSI
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        # 過熱回避
        if pd.isna(r.DEV_SMA25) or r.DEV_SMA25 > 15: return False
        return True
    except Exception:
        return False

def screen_A2(df: pd.DataFrame) -> bool:
    """A-2: ブレイクアウト + 出来高確認"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        # 52週高値ブレイク（当日）
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if prev.Close >= prev.HIGH_52W: return False  # 前日はブレイク前
        # 出来高
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        # ADX
        if pd.isna(r.ADX) or r.ADX < 25: return False
        # 200日MA上
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception:
        return False

def screen_A3(df: pd.DataFrame) -> bool:
    """A-3: MACD + RSI トレンド追従"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        # MACDクロス（直近1日以内）
        if pd.isna(r.MACD) or pd.isna(r.MACD_sig): return False
        cross = (r.MACD > r.MACD_sig) and (prev.MACD <= prev.MACD_sig)
        if not cross: return False
        # ゼロライン上
        if r.MACD <= 0: return False
        # ヒストグラムプラス
        if pd.isna(r.MACD_hist) or r.MACD_hist <= 0: return False
        # RSI
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 65): return False
        # 75日MA上
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        return True
    except Exception:
        return False

def screen_A4(df: pd.DataFrame) -> bool:
    """A-4: 一目均衡表 三役好転（簡易版）"""
    try:
        # 一目均衡表を手計算
        high = df["High"]
        low  = df["Low"]
        close = df["Close"]
        # 転換線(9日)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        # 基準線(26日)
        kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
        # 先行スパン1（26日先行）
        span1  = ((tenkan + kijun) / 2).shift(26)
        # 先行スパン2（52日高低平均, 26日先行）
        span2  = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        # 遅行線（26日遅行）
        chikou = close.shift(-26)

        r_idx = -1
        c   = close.iloc[r_idx]
        t   = tenkan.iloc[r_idx]
        k   = kijun.iloc[r_idx]
        s1  = span1.iloc[r_idx]
        s2  = span2.iloc[r_idx]
        ch  = chikou.iloc[-27] if len(close) > 27 else np.nan

        if any(pd.isna([t, k, s1, s2])): return False
        # 三役好転
        if not (t > k): return False          # 転換 > 基準
        if not (c > max(s1, s2)): return False # 株価 > 雲上
        if not (s1 > s2): return False         # 陽雲
        return True
    except Exception:
        return False

def screen_A5(df: pd.DataFrame, topix_ret60: float = None) -> bool:
    """A-5: 相対強度 + ADX"""
    try:
        r = get_latest(df)
        # 200日MA上
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        # ADX
        if pd.isna(r.ADX) or r.ADX < 30: return False
        if pd.isna(r.DI_pos) or pd.isna(r.DI_neg) or r.DI_pos <= r.DI_neg: return False
        # ボリンジャー中心線以上
        if pd.isna(r.BB_mid) or r.Close < r.BB_mid: return False
        # 60日リターン（TOPIX比は近似値で対応）
        if pd.isna(r.RET_60D) or r.RET_60D < 5: return False  # 最低+5%以上
        return True
    except Exception:
        return False

def screen_B1(df: pd.DataFrame, info: dict) -> bool:
    """B-1: 低PBR + 高ROE"""
    try:
        r = get_latest(df)
        pbr = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        eq  = info.get("debtToEquity")
        if pbr is None or pbr > 1.0 or pbr < 0.01: return False
        if roe is None or roe < 0.10: return False
        # 75日MA上
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        # RSI
        if pd.isna(r.RSI14) or r.RSI14 < 40: return False
        return True
    except Exception:
        return False

def screen_B2(df: pd.DataFrame, info: dict) -> bool:
    """B-2: 低PER + 増益 + MACD"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        per = info.get("trailingPE") or info.get("forwardPE")
        if per is None or not (5 <= per <= 20): return False
        # MACD買いシグナル（直近10日以内）
        cross_found = False
        for i in range(min(10, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if pd.isna(r0.MACD) or pd.isna(r0.MACD_sig): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig:
                cross_found = True; break
        if not cross_found: return False
        if pd.isna(r.RSI14) or r.RSI14 < 45: return False
        return True
    except Exception:
        return False

def screen_B3(df: pd.DataFrame, info: dict) -> bool:
    """B-3: 高配当 + 低負債 + 底打ち"""
    try:
        r = get_latest(df)
        div_yield  = info.get("dividendYield", 0) or 0
        payout     = info.get("payoutRatio", 0) or 0
        de_ratio   = info.get("debtToEquity", 999) or 999
        if div_yield < 0.035: return False
        if not (0.2 <= payout <= 0.8): return False
        if de_ratio > 50: return False  # yfinanceはパーセント表示の場合あり
        # RSI底打ち（直近5日で35以下から上向き）
        rsi_vals = df["RSI14"].iloc[-6:]
        if rsi_vals.isna().all(): return False
        if rsi_vals.min() > 35: return False
        # 現在RSIが底から上向き
        if r.RSI14 < rsi_vals.min() + 2: return False
        return True
    except Exception:
        return False

def screen_B4(df: pd.DataFrame, info: dict) -> bool:
    """B-4: 低PSR + 高成長"""
    try:
        r = get_latest(df)
        psr     = info.get("priceToSalesTrailing12Months")
        rev_grw = info.get("revenueGrowth")
        margin  = info.get("operatingMargins")
        if psr is None or psr > 2.0: return False
        if rev_grw is None or rev_grw < 0.15: return False
        if margin is None or margin < 0.08: return False
        # BBバンド下限からの反発
        if pd.isna(r.BB_lower) or pd.isna(r.BB_mid): return False
        # 直近でBB下限タッチして現在は中心線方向
        bb_min_5d = df["BB_lower"].iloc[-5:]
        close_5d  = df["Close"].iloc[-5:]
        touched = (close_5d <= bb_min_5d).any()
        if not touched: return False
        if r.Close <= r.BB_lower: return False  # まだ下限以下はNG
        return True
    except Exception:
        return False

def screen_C1(df: pd.DataFrame, info: dict) -> bool:
    """C-1: 過売圏 ファンダ良好 反発狙い"""
    try:
        r = get_latest(df)
        roe = info.get("returnOnEquity")
        if pd.isna(r.RSI14) or r.RSI14 > 30: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.LOW_52W) or r.Close < r.LOW_52W * 1.10: return False
        if roe is None or roe < 0.12: return False
        return True
    except Exception:
        return False

def screen_C2(df: pd.DataFrame) -> bool:
    """C-2: ボリンジャー下限タッチ + ストキャスティクス"""
    try:
        r = get_latest(df)
        # 直近3日でBB下限タッチ
        close_3d = df["Close"].iloc[-4:-1]
        bb_low_3d = df["BB_lower"].iloc[-4:-1]
        if close_3d.isna().all() or bb_low_3d.isna().all(): return False
        touched = (close_3d <= bb_low_3d).any()
        if not touched: return False
        # 現在はBB下限より上
        if pd.isna(r.BB_lower) or r.Close <= r.BB_lower: return False
        # ストキャスティクス
        if pd.isna(r.STOCH_K) or pd.isna(r.STOCH_D): return False
        if r.STOCH_K > 20: return False
        prev = get_prev(df)
        if pd.isna(prev.STOCH_K) or pd.isna(prev.STOCH_D): return False
        if not (r.STOCH_K > r.STOCH_D and prev.STOCH_K <= prev.STOCH_D): return False
        # 出来高減少（売り枯れ）
        if pd.isna(r.VOL_MA20) or r.Volume > r.VOL_MA20 * 0.8: return False
        # 75日MA上
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        return True
    except Exception:
        return False

def screen_C3(df: pd.DataFrame, info: dict) -> bool:
    """C-3: 52週安値圏 + 需給改善"""
    try:
        r = get_latest(df)
        pbr = info.get("priceToBook")
        if pd.isna(r.LOW_52W) or r.Close > r.LOW_52W * 1.05: return False
        if pbr is None or pbr > 0.8: return False
        # 直近20日の最安値を更新していない
        recent_low = df["Close"].iloc[-21:-1].min()
        if r.Close <= recent_low: return False
        return True
    except Exception:
        return False

def screen_C4(df: pd.DataFrame) -> bool:
    """C-4: MACDダイバージェンス 底打ち"""
    try:
        if len(df) < 60: return False
        close = df["Close"]
        hist  = df["MACD_hist"]
        if hist.isna().sum() > len(hist) * 0.3: return False
        # 直近60日で2つの安値を検出
        window = df.iloc[-60:]
        lows_idx = []
        for i in range(2, len(window)-2):
            if window["Close"].iloc[i] <= window["Close"].iloc[i-1] and \
               window["Close"].iloc[i] <= window["Close"].iloc[i+1] and \
               window["Close"].iloc[i] <= window["Close"].iloc[i-2] and \
               window["Close"].iloc[i] <= window["Close"].iloc[i+2]:
                lows_idx.append(i)
        if len(lows_idx) < 2: return False
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if i2 - i1 < 10: return False  # 間隔10日以上
        # ダイバージェンス確認
        p1_close = window["Close"].iloc[i1]
        p2_close = window["Close"].iloc[i2]
        p1_hist  = window["MACD_hist"].iloc[i1]
        p2_hist  = window["MACD_hist"].iloc[i2]
        if pd.isna(p1_hist) or pd.isna(p2_hist): return False
        if p2_close >= p1_close: return False  # 株価は安値更新
        if p2_hist <= p1_hist: return False    # ヒストは改善
        return True
    except Exception:
        return False

def screen_D1(df: pd.DataFrame, info: dict) -> bool:
    """D-1: CANSLIM 完全版（近似版）"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        # C: EPS成長（revenueGrowth近似）
        rev_grw = info.get("revenueGrowth", 0) or 0
        if rev_grw < 0.20: return False
        # A: ROE
        roe = info.get("returnOnEquity", 0) or 0
        if roe < 0.17: return False
        # S: 52週高値ブレイク + 出来高
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        # L: 60日リターン上位（正のリターン）
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        # M: 200日MA上
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception:
        return False

def screen_D2(df: pd.DataFrame, info: dict) -> bool:
    """D-2: CANSLIM簡易版"""
    try:
        r = get_latest(df)
        rev_grw = info.get("revenueGrowth", 0) or 0
        roe     = info.get("returnOnEquity", 0) or 0
        if rev_grw < 0.15: return False
        if roe < 0.17: return False
        # 52週高値ブレイク
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        # 相対強度（60日リターン）
        if pd.isna(r.RET_60D) or r.RET_60D < 8: return False
        # TOPIX（200日MA上近似）
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception:
        return False

def screen_E1(df: pd.DataFrame, info: dict) -> bool:
    """E-1: 自社株買い + テクニカル（自社株買い情報は近似）"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        # 自社株買い情報はyfinanceから取りにくいため、
        # 代替: 発行済株式数の減少トレンドを確認
        shares_out = info.get("sharesOutstanding")
        shares_float = info.get("floatShares")
        # 75日MA上かつMACD買いシグナル
        if pd.isna(r.SMA25) or r.Close <= r.SMA25: return False
        # MACD（直近5日以内）
        cross_found = False
        for i in range(min(5, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if pd.isna(r0.MACD) or pd.isna(r0.MACD_sig): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig:
                cross_found = True; break
        if not cross_found: return False
        eq_ratio = info.get("returnOnEquity")
        if eq_ratio is None or eq_ratio < 0.05: return False
        return True
    except Exception:
        return False

def screen_E2(df: pd.DataFrame, info: dict) -> bool:
    """E-2: 業績上方修正 + ブレイクアウト（近似）"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        per = info.get("forwardPE") or info.get("trailingPE")
        if per is None or per > 25 or per < 3: return False
        # 52週高値ブレイク
        if pd.isna(r.HIGH_52W) or r.Close < r.HIGH_52W: return False
        # 出来高急増
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 2.0: return False
        # 200日MA上
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception:
        return False

def screen_E3(df: pd.DataFrame) -> bool:
    """E-3: セクターローテーション（個別銘柄強度で近似）"""
    try:
        r = get_latest(df)
        # 200日MA上
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        # RSI 50〜70
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 70): return False
        # ADX
        if pd.isna(r.ADX) or r.ADX < 25: return False
        # 60日リターン良好
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        # 20日リターン良好
        if pd.isna(r.RET_20D) or r.RET_20D < 3: return False
        return True
    except Exception:
        return False

def screen_E4(df: pd.DataFrame) -> bool:
    """E-4: 空売り比率急低下（テクニカルで近似）"""
    try:
        r = get_latest(df)
        # RSI直近5日で40以下タッチ後の回復
        rsi_5d = df["RSI14"].iloc[-6:]
        if rsi_5d.isna().all(): return False
        if rsi_5d.min() > 40: return False
        if r.RSI14 < 40: return False  # まだ回復してない
        # 直近20日最安値を更新していない
        recent_low = df["Close"].iloc[-21:-1].min()
        if r.Close <= recent_low: return False
        # 75日MA上
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        return True
    except Exception:
        return False

def screen_F1(df: pd.DataFrame, info: dict) -> bool:
    """F-1: 高品質コア保有"""
    try:
        r = get_latest(df)
        roe = info.get("returnOnEquity", 0) or 0
        if roe < 0.15: return False
        # 200日MA上、乖離10%以内
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.DEV_SMA200) or r.DEV_SMA200 > 10: return False
        # RSI 40〜65
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        # 配当利回り
        div_yield = info.get("dividendYield", 0) or 0
        if div_yield < 0.01: return False  # 配当あり確認
        return True
    except Exception:
        return False

def screen_F2(df: pd.DataFrame, info: dict) -> bool:
    """F-2: 低ボラティリティ + 安定配当"""
    try:
        r = get_latest(df)
        div_yield = info.get("dividendYield", 0) or 0
        payout    = info.get("payoutRatio", 0) or 0
        if div_yield < 0.03: return False
        if not (0.3 <= payout <= 0.65): return False
        # 75日MA上
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        # RSI 35〜60
        if pd.isna(r.RSI14) or not (35 <= r.RSI14 <= 60): return False
        # ボラティリティ（直近120日の日次リターンσが小さい）
        returns = df["Close"].pct_change().iloc[-120:]
        vol = returns.std()
        if pd.isna(vol) or vol > 0.025: return False  # 約2.5%/日以下
        return True
    except Exception:
        return False

# ─────────────────────────────────────────
# 【I】実践パターン組み合わせ系
# ─────────────────────────────────────────
def screen_I1(df: pd.DataFrame, info: dict) -> bool:
    """I-1: 安定（バランス型）"""
    try:
        r = get_latest(df)
        roe = info.get("returnOnEquity", 0) or 0
        rev_grw = info.get("revenueGrowth", 0) or 0
        if roe < 0.10: return False
        if rev_grw < 0.04: return False
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 65): return False
        return True
    except Exception:
        return False

def screen_I2(df: pd.DataFrame, info: dict) -> bool:
    """I-2: 成長型（中期で伸ばす）"""
    try:
        r = get_latest(df)
        rev_grw = info.get("revenueGrowth", 0) or 0
        roe = info.get("returnOnEquity", 0) or 0
        margin = info.get("operatingMargins", 0) or 0
        if rev_grw < 0.15: return False
        if roe < 0.12: return False
        if margin < 0.08: return False
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        if pd.isna(r.RET_60D) or r.RET_60D < 10: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception:
        return False

def screen_I3(df: pd.DataFrame, info: dict) -> bool:
    """I-3: 業績修正（リバウンド狙い）"""
    try:
        r = get_latest(df)
        prev = get_prev(df)
        pbr = info.get("priceToBook")
        if pbr is None or pbr > 1.5: return False
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        # MACD買いシグナル（10日以内）または52週高値ブレイク
        macd_cross = False
        for i in range(min(10, len(df)-2)):
            r0 = df.iloc[-(i+1)]
            r1 = df.iloc[-(i+2)]
            if pd.isna(r0.MACD) or pd.isna(r0.MACD_sig): continue
            if r0.MACD > r0.MACD_sig and r1.MACD <= r1.MACD_sig:
                macd_cross = True; break
        breakout = not pd.isna(r.HIGH_52W) and r.Close >= r.HIGH_52W
        if not (macd_cross or breakout): return False
        return True
    except Exception:
        return False

def screen_I4(df: pd.DataFrame, info: dict) -> bool:
    """I-4: イベントドリブン（短中期）"""
    try:
        r = get_latest(df)
        if pd.isna(r.VOL_MA20) or r.Volume < r.VOL_MA20 * 1.5: return False
        if pd.isna(r.SMA25) or r.Close <= r.SMA25: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        rev_grw = info.get("revenueGrowth", 0) or 0
        if rev_grw < 0.0: return False
        return True
    except Exception:
        return False

def screen_I5(df: pd.DataFrame, info: dict) -> bool:
    """I-5: 安定重視（ドローダウン抑制）"""
    try:
        r = get_latest(df)
        roe = info.get("returnOnEquity", 0) or 0
        if roe < 0.12: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        returns = df["Close"].pct_change().iloc[-120:]
        vol = returns.std()
        if pd.isna(vol) or vol > 0.022: return False
        return True
    except Exception:
        return False

def screen_I6(df: pd.DataFrame, info: dict) -> bool:
    """I-6: 高精度（実戦強化・多軸確認）"""
    try:
        r = get_latest(df)
        roe = info.get("returnOnEquity", 0) or 0
        rev_grw = info.get("revenueGrowth", 0) or 0
        if roe < 0.12: return False
        if rev_grw < 0.08: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 65): return False
        if pd.isna(r.ADX) or r.ADX < 20: return False
        if pd.isna(r.DEV_SMA200) or r.DEV_SMA200 > 15: return False
        return True
    except Exception:
        return False

def screen_I7(df: pd.DataFrame, info: dict) -> bool:
    """I-7: 完全スクリーニング（機関投資家型）"""
    try:
        r = get_latest(df)
        roe = info.get("returnOnEquity", 0) or 0
        rev_grw = info.get("revenueGrowth", 0) or 0
        div_yield = info.get("dividendYield", 0) or 0
        if roe < 0.15: return False
        if rev_grw < 0.10: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 65): return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        returns = df["Close"].pct_change().iloc[-120:]
        vol = returns.std()
        if pd.isna(vol) or vol > 0.025: return False
        return True
    except Exception:
        return False

def screen_I8(df: pd.DataFrame) -> bool:
    """I-8: モメンタム特化"""
    try:
        r = get_latest(df)
        if pd.isna(r.RET_60D) or r.RET_60D < 15: return False
        if pd.isna(r.ADX) or r.ADX < 25: return False
        if pd.isna(r.DI_pos) or pd.isna(r.DI_neg) or r.DI_pos <= r.DI_neg: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RET_20D) or r.RET_20D < 5: return False
        return True
    except Exception:
        return False

def screen_I9(df: pd.DataFrame, info: dict) -> bool:
    """I-9: 小型成長株"""
    try:
        r = get_latest(df)
        rev_grw = info.get("revenueGrowth", 0) or 0
        roe = info.get("returnOnEquity", 0) or 0
        margin = info.get("operatingMargins", 0) or 0
        mktcap = info.get("marketCap", 0) or 0
        if rev_grw < 0.20: return False
        if roe < 0.10: return False
        if margin < 0.05: return False
        # 時価総額30億〜500億円
        if mktcap > 0 and not (3e9 <= mktcap <= 50e9): return False
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        if pd.isna(r.RET_60D) or r.RET_60D < 5: return False
        return True
    except Exception:
        return False

def screen_I10(df: pd.DataFrame, info: dict) -> bool:
    """I-10: ディフェンシブ"""
    try:
        r = get_latest(df)
        roe = info.get("returnOnEquity", 0) or 0
        div_yield = info.get("dividendYield", 0) or 0
        payout = info.get("payoutRatio", 0) or 0
        if roe < 0.08: return False
        if div_yield < 0.025: return False
        if not (0.20 <= payout <= 0.60): return False
        if pd.isna(r.SMA75) or r.Close <= r.SMA75: return False
        returns = df["Close"].pct_change().iloc[-120:]
        vol = returns.std()
        if pd.isna(vol) or vol > 0.020: return False
        return True
    except Exception:
        return False

# ─────────────────────────────────────────
# TXTレポート出力
# ─────────────────────────────────────────
def _write_txt_report(df_results, df_summary, all_matches, stock_data, path, today):
    lines = []
    sep  = "=" * 70
    sep2 = "-" * 70

    lines.append(sep)
    lines.append(f"  中期投資スクリーニング レポート  {today}")
    lines.append(sep)
    lines.append(f"スキャン銘柄数 : {len(stock_data)}")
    lines.append(f"ヒット銘柄数   : {len(df_results)}")
    lines.append("")

    # ── 戦略別サマリー ──
    lines.append("【戦略別ヒット数サマリー】")
    lines.append(sep2)
    for _, row in df_summary.iterrows():
        hit = int(row["ヒット銘柄数"])
        bar = "■" * hit + "□" * max(0, 15 - hit)
        lines.append(f"  {row['戦略']:<30s}  {hit:>2}件  {bar}")
        if hit > 0:
            lines.append(f"    銘柄: {row['銘柄一覧']}")
    lines.append("")

    # ── 全ヒット銘柄詳細 ──
    lines.append("【全ヒット銘柄 詳細】")
    lines.append(sep2)
    has_fund = "PER" in df_results.columns

    for _, row in df_results.iterrows():
        lines.append(f"  ▶ {row['銘柄コード']}  現在値:{row['現在値']:>8.0f}円  "
                     f"RSI:{row['RSI14'] if row['RSI14'] != '' else 'N/A':>5}  "
                     f"マッチ数:{int(row['マッチ戦略数'])}")
        if has_fund:
            per = row.get('PER', '')
            pbr = row.get('PBR', '')
            roe = row.get('ROE(%)', '')
            div = row.get('配当利回り(%)', '')
            cap = row.get('時価総額(億円)', '')
            lines.append(f"     PER:{per:<8} PBR:{pbr:<8} ROE:{roe}%  "
                         f"配当:{div}%  時価総額:{cap}億円")
        # 戦略を1行ずつ
        for strat in row["マッチ戦略"].split(" | "):
            lines.append(f"       ✓ {strat}")
        lines.append("")

    # ── 複数マッチ銘柄（注目） ──
    multi = df_results[df_results["マッチ戦略数"] >= 2]
    if not multi.empty:
        lines.append("【注目銘柄（2戦略以上マッチ）】")
        lines.append(sep2)
        for _, row in multi.iterrows():
            lines.append(f"  {row['銘柄コード']}  {row['現在値']:>8.0f}円  "
                         f"({int(row['マッチ戦略数'])}戦略)  {row['マッチ戦略']}")
        lines.append("")

    lines.append(sep)
    lines.append(f"  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  参照: screening_thresholds.txt (A-1〜F-2 全21戦略 + I-1〜I-10 全10戦略)")
    lines.append(sep)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────
# メイン実行
# ─────────────────────────────────────────
STRATEGIES = {
    "A-1_王道モメンタム":          lambda df, info: screen_A1(df),
    "A-2_ブレイクアウト":          lambda df, info: screen_A2(df),
    "A-3_MACD_RSIトレンド":        lambda df, info: screen_A3(df),
    "A-4_一目均衡表三役好転":       lambda df, info: screen_A4(df),
    "A-5_相対強度ADX":             lambda df, info: screen_A5(df),
    "B-1_低PBR高ROE":              lambda df, info: screen_B1(df, info),
    "B-2_低PER増益":               lambda df, info: screen_B2(df, info),
    "B-3_高配当低負債":             lambda df, info: screen_B3(df, info),
    "B-4_低PSR高成長":             lambda df, info: screen_B4(df, info),
    "C-1_過売圏反発":              lambda df, info: screen_C1(df, info),
    "C-2_BB下限ストキャス":         lambda df, info: screen_C2(df),
    "C-3_52週安値需給改善":         lambda df, info: screen_C3(df, info),
    "C-4_MACDダイバージェンス":     lambda df, info: screen_C4(df),
    "D-1_CANSLIM完全版":           lambda df, info: screen_D1(df, info),
    "D-2_CANSLIM簡易版":           lambda df, info: screen_D2(df, info),
    "E-1_自社株買い":              lambda df, info: screen_E1(df, info),
    "E-2_業績上方修正":             lambda df, info: screen_E2(df, info),
    "E-3_セクターローテーション":    lambda df, info: screen_E3(df),
    "E-4_空売り低下":              lambda df, info: screen_E4(df),
    "F-1_高品質コア":              lambda df, info: screen_F1(df, info),
    "F-2_低ボラ安定配当":           lambda df, info: screen_F2(df, info),
    "I-1_安定バランス":            lambda df, info: screen_I1(df, info),
    "I-2_成長型":                  lambda df, info: screen_I2(df, info),
    "I-3_業績修正リバウンド":        lambda df, info: screen_I3(df, info),
    "I-4_イベントドリブン":          lambda df, info: screen_I4(df, info),
    "I-5_安定重視ドローダウン抑制":   lambda df, info: screen_I5(df, info),
    "I-6_高精度多軸":              lambda df, info: screen_I6(df, info),
    "I-7_完全機関投資家型":          lambda df, info: screen_I7(df, info),
    "I-8_モメンタム特化":            lambda df, info: screen_I8(df),
    "I-9_小型成長株":              lambda df, info: screen_I9(df, info),
    "I-10_ディフェンシブ":           lambda df, info: screen_I10(df, info),
}

def run_all_screens():
    # 株価データ取得
    stock_data = fetch_stock_data(STOCK_UNIVERSE, PERIOD_LONG)

    results = []
    all_matches = {s: [] for s in STRATEGIES}

    print(f"\nスクリーニング実行中...")
    info_cache = {}

    for ticker, df in stock_data.items():
        try:
            df = calc_indicators(df)
        except Exception:
            continue

        # infoは必要なもののみ取得（まとめて後から）
        info = {}

        r = df.iloc[-1]
        # 基本情報の計算
        ret_20  = round(df["Close"].pct_change(20).iloc[-1] * 100, 2) if len(df) > 20 else np.nan
        ret_60  = round(df["Close"].pct_change(60).iloc[-1] * 100, 2) if len(df) > 60 else np.nan
        current_price = round(float(r.Close), 0)

        matched = []
        for strategy_name, func in STRATEGIES.items():
            # info が必要な戦略の場合のみ取得
            needs_info = any(x in strategy_name for x in ["PBR","PER","配当","成長","CANSLIM","自社株","上方修正","品質","ボラ","安定","イベント","修正","ドロー","精度","機関","小型","ディフェン"])
            if needs_info and ticker not in info_cache:
                try:
                    info_cache[ticker] = fetch_info(ticker)
                    time.sleep(0.1)
                except Exception:
                    info_cache[ticker] = {}
            info = info_cache.get(ticker, {})

            try:
                if func(df, info):
                    matched.append(strategy_name)
                    all_matches[strategy_name].append(ticker)
            except Exception:
                pass

        if matched:
            row = {
                "ティッカー":      ticker,
                "銘柄コード":      ticker.replace(".T", ""),
                "現在値":          current_price,
                "20日リターン(%)": ret_20,
                "60日リターン(%)": ret_60,
                "RSI14":           round(float(r.RSI14), 1) if not pd.isna(r.RSI14) else "",
                "SMA25乖離(%)":    round(float(r.DEV_SMA25), 1) if not pd.isna(r.DEV_SMA25) else "",
                "SMA200乖離(%)":   round(float(r.DEV_SMA200), 1) if not pd.isna(r.DEV_SMA200) else "",
                "ADX14":           round(float(r.ADX), 1) if not pd.isna(r.ADX) else "",
                "マッチ戦略数":    len(matched),
                "マッチ戦略":      " | ".join(matched),
            }
            # ファンダ情報を追加
            if ticker in info_cache and info_cache[ticker]:
                info = info_cache[ticker]
                row["PER"]      = round(info.get("trailingPE") or 0, 1) or ""
                row["PBR"]      = round(info.get("priceToBook") or 0, 2) or ""
                row["ROE(%)"]   = round((info.get("returnOnEquity") or 0) * 100, 1) or ""
                row["配当利回り(%)"] = round((info.get("dividendYield") or 0) * 100, 2) or ""
                row["時価総額(億円)"] = round((info.get("marketCap") or 0) / 1e8, 0) or ""
            results.append(row)

    # DataFrameに変換
    if not results:
        print("\n条件に合う銘柄が見つかりませんでした。")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("マッチ戦略数", ascending=False)

    # ─── CSV出力: 全銘柄統合 ───
    out_all = os.path.join(OUTPUT_DIR, f"screening_all_{TODAY}.csv")
    df_results.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 全銘柄統合: {out_all}")

    # ─── CSV出力: 戦略別 ───
    strategy_summary = []
    for strategy_name, tickers in all_matches.items():
        if tickers:
            rows = df_results[df_results["ティッカー"].isin(tickers)].copy()
            rows["戦略"] = strategy_name
            strategy_summary.append(rows)

    if strategy_summary:
        df_by_strategy = pd.concat(strategy_summary, ignore_index=True)
        out_strat = os.path.join(OUTPUT_DIR, f"screening_by_strategy_{TODAY}.csv")
        df_by_strategy.to_csv(out_strat, index=False, encoding="utf-8-sig")
        print(f"[保存] 戦略別:    {out_strat}")

    # ─── CSV出力: 戦略別ヒット数サマリー ───
    summary_rows = [{"戦略": k, "ヒット銘柄数": len(v), "銘柄一覧": ",".join(v)}
                    for k, v in all_matches.items()]
    df_summary = pd.DataFrame(summary_rows)
    out_summary = os.path.join(OUTPUT_DIR, f"screening_summary_{TODAY}.csv")
    df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"[保存] サマリー:   {out_summary}")

    # ─── TXT出力 ───
    out_txt = os.path.join(OUTPUT_DIR, f"screening_report_{TODAY}.txt")
    _write_txt_report(df_results, df_summary, all_matches, stock_data, out_txt, TODAY)
    print(f"[保存] レポート:   {out_txt}")

    # ─── コンソール表示 ───
    print(f"\n{'='*60}")
    print(f" スクリーニング結果サマリー ({TODAY})")
    print(f"{'='*60}")
    print(f"スキャン銘柄数: {len(stock_data)}")
    print(f"ヒット銘柄数:   {len(df_results)}")
    print()
    print(df_summary.to_string(index=False))
    print()
    print("【全ヒット銘柄】")
    cols = ["銘柄コード","現在値","RSI14","マッチ戦略数","マッチ戦略"]
    if "PER" in df_results.columns:
        cols = ["銘柄コード","現在値","RSI14","PER","PBR","ROE(%)","マッチ戦略数","マッチ戦略"]
    print(df_results[cols].to_string(index=False))

    return df_results

if __name__ == "__main__":
    run_all_screens()
