# mia-rag

20冊のマルクス経済学古典（Marxists Internet Archive）を対象にした
RAG（検索拡張生成）リーダーです。Tkinter のデスクトップ UI と CLI を同梱します。

- 検索: BM25（SQLite FTS5）＋ ベクトル検索（`nomic-embed-text`）のハイブリッド
- 生成: OpenCode Go（`deepseek-v4.1-flash`）
- 出典: 回答中の `[1]` などの番号と、Marxists Internet Archive の URL を表示

コーパス本文・埋め込み・索引はリポジトリに含まれません。手元のテキストから初回に構築します。

## 必要なもの

- Python 3.10 以上（Tkinter 同梱の配布物。Linux では `python3-tk`）
- [Ollama](https://ollama.com/) と埋め込みモデル: `ollama pull nomic-embed-text`
- OpenCode Go の API キー（環境変数 `OPENCODE_API_KEY`、または OpenCode の
  `~/.local/share/opencode/auth.json`、または GUI の Settings で入力）

## uvx で起動（インストール不要）

```bash
uvx --from git+https://github.com/vbe-robotron/mia-rag mia-rag-gui
```

CLI:

```bash
uvx --from git+https://github.com/vbe-robotron/mia-rag mia-rag "帝国主義と金融資本の関係は？"
```

常設インストール:

```bash
uv tool install --from git+https://github.com/vbe-robotron/mia-rag mia-rag
```

## 初回セットアップ

1. コーパス（`----- SOURCE:` マーカー付きの UTF-8 `.txt` 20冊）を任意のフォルダに置く
2. GUI を起動し、**Settings** で *Corpus folder* と *Index folder*、API キーを設定
3. **Rebuild index** を押す（チャンク化 → 埋め込み → FTS 構築）

CPU のみの場合、埋め込み（約2万チャンク）に30〜60分かかります。
索引は再構築不要です。

CLI で構築する場合:

```bash
mia-rag --build --corpus /path/to/text --data-dir ~/.local/share/mia-rag
mia-rag --build --no-embed ...        # 埋め込みを後回しにする場合
```

## 使い方

GUI:

```bash
mia-rag-gui
```

- 質問を入力して **Ask**（`Ctrl+Enter` でも実行）
- **Books** で対象文献を絞り込み（未選択なら全冊）
- **Sources** タブの行をダブルクリックで出典 URL を開く

CLI:

```bash
mia-rag "資本の原始的蓄積における暴力の役割は？"
mia-rag --k 12 --books 01,17 "ルクセンブルクの蓄積論の要点は？" --show
```

データは既定で `~/.local/share/mia-rag/`、設定は
`~/.config/mia-rag/settings.json`（0600）に保存されます。

## 構成

```
src/mia_rag/
  corpus.py   テキストのチャンク化
  embed.py    Ollama 埋め込みと索引生成
  index.py    FTS5 / ベクトル検索 / RRF 融合
  api.py      OpenCode Go クライアント
  rag.py      クエリ展開と回答生成
  cli.py      コマンドライン
  gui.py      Tkinter UI
```

## 出典・ライセンス

- 本文は [Marxists Internet Archive](https://www.marxists.org/) の公開テキストに由来します。
  再配布時は同アーカイブへのクレジットが必要です。
- 本リポジトリのコードは MIT License（`LICENSE`）。
- `nomic-embed-text` および `deepseek-v4.1-flash` は各提供元の条件に従います。
