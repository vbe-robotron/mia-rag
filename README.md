# mia-rag

20冊のマルクス経済学古典（Marxists Internet Archive）を対象にした
RAG（検索拡張生成）リーダーです。ローカルWeb UIとCLIを同梱します。

- 検索: BM25（SQLite FTS5）＋ ベクトル検索（`nomic-embed-text`）のハイブリッド
- 生成: OpenCode Go（`deepseek-v4.1-flash`）。回答はストリーミング表示
- 出典: 回答中の `[1]` などの番号と、Marxists Internet Archive の URL を表示
- 設定: APIキー・モデル・コーパス/索引フォルダをすべてブラウザ上のGUIから設定

コーパス本文・埋め込み・索引はリポジトリに含まれません。手元のテキストから初回に構築します。

## 必要なもの

- Python 3.10 以上
- [Ollama](https://ollama.com/) と埋め込みモデル: `ollama pull nomic-embed-text`
- OpenCode Go の API キー（GUIの Settings で入力、または環境変数
  `OPENCODE_API_KEY`、または OpenCode の `~/.local/share/opencode/auth.json`）

## uvx で起動（インストール不要）

```bash
uvx --from git+https://github.com/vbe-robotron/mia-rag mia-rag-web
```

ブラウザで `http://127.0.0.1:8765/` が開きます。ポート変更は `--port 9000`、
ブラウザを開かない場合は `--no-browser`。

CLI:

```bash
uvx --from git+https://github.com/vbe-robotron/mia-rag mia-rag "帝国主義と金融資本の関係は？"
```

常設インストール:

```bash
uv tool install --from git+https://github.com/vbe-robotron/mia-rag mia-rag
mia-rag-web
```

## 初回セットアップ（GUI）

1. **Settings** を開く
2. *Corpus folder* に20冊の `.txt` があるフォルダ、*Index folder* に索引の保存先を指定
3. *OpenCode Go API key* を入力して **Save**
4. **Rebuild index** → **Start**（チャンク化 → 埋め込み → FTS 構築）

CPUのみの場合、埋め込み（約2万チャンク）に30〜60分かかります。索引は再構築不要です。

CLI で構築する場合:

```bash
mia-rag --build --corpus /path/to/text --data-dir ~/.local/share/mia-rag
mia-rag --build --no-embed ...        # 埋め込みを後回しにする場合
```

## 使い方

- 質問を入力して **Ask**（`Ctrl/⌘ + Enter`）
- 左の **Books** で対象文献を絞り込み（未選択なら全20冊）
- 回答中の `[1]` をクリックすると出典へ移動。**Sources** の URL から原文を開けます
- **New** で会話をクリア、**Stop** で生成を中断

CLI:

```bash
mia-rag "資本の原始的蓄積における暴力の役割は？"
mia-rag --k 12 --books 01,17 "ルクセンブルクの蓄積論の要点は？" --show
```

設定は `~/.config/mia-rag/settings.json`（0600）、索引は既定で
`~/.local/share/mia-rag/` に保存されます。Web UI は `127.0.0.1` のみにバインドされ、
APIキーはブラウザへ返送されません。

## 構成

```
src/mia_rag/
  corpus.py     テキストのチャンク化
  embed.py      Ollama 埋め込みと索引生成
  index.py      FTS5 / ベクトル検索 / RRF 融合
  api.py        OpenCode Go クライアント（ストリーミング対応）
  rag.py        クエリ展開と回答生成
  cli.py        コマンドライン
  web.py        FastAPI サーバ (SSE)
  static/       Web UI (HTML/CSS/JS)
```

## 出典・ライセンス

- 本文は [Marxists Internet Archive](https://www.marxists.org/) の公開テキストに由来します。
  再配布時は同アーカイブへのクレジットが必要です。
- 本リポジトリのコードは MIT License（`LICENSE`）。
- `nomic-embed-text` および `deepseek-v4.1-flash` は各提供元の条件に従います。
