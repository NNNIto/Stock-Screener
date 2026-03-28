# スクリーナー実行 → GitHub プッシュ

以下の手順をすべて実行してください。

## 手順

### 1. スクリーナー実行
```
cd /home/sys1/Stock-Screener && python3 screener.py
```
実行中はログを表示し、エラーがあれば報告して停止してください。

### 2. git add
以下のみをステージングしてください（`.env` と `__pycache__` は**絶対に含めない**）:
- `screener.py`
- `.gitignore`
- `results/` 以下の最新フォルダ（直近1件、`YYYY-MM-DD_HHMM` 形式）
- 削除済みの古い results フォルダ（`git rm` で反映）

### 3. git commit
コミットメッセージは以下の形式:
```
スクリーニング結果: YYYY-MM-DD HH:MM
```
今日の日付と現在時刻を使ってください。

### 4. git push
```
git push origin main
```

完了後、GitHubにプッシュされたコミットのハッシュとURLを報告してください。
