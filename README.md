# legacy-csl

`legacy-csl` provides reproducible reference implementations of two
categorical cross-situational learning models. **Single-CSL** learns a
word–meaning system from one agent's multimodal observations and fixed
utterances. **Inter-CSL** lets two agents learn through a Metropolis–Hastings
naming game and can predict one agent's observations from the other's words.

`legacy-csl` は、カテゴリカル交差状況学習モデルを再現可能な形で実行する
ためのリファレンス実装です。**Single-CSL** は1エージェントのマルチモーダル
観測と固定発話から単語と意味の対応を学習します。**Inter-CSL** は2エージェント
間の Metropolis–Hastings naming game によって記号体系を学習し、相手の単語から
観測を予測できます。

The package includes strict YAML configuration, explicit random seeds,
checkpointing, legacy-compatible CSV export, standard metrics, and compact
plots. It can read either the included synthetic-data format or the processed
legacy datadump layout.

本パッケージには、厳格な YAML 設定、明示的乱数シード、チェックポイント、
主要な legacy 互換 CSV 出力、標準評価指標、基本的な可視化が含まれます。
同梱の合成データ形式と、処理済み legacy datadump 形式の両方を読み込めます。

## Supported model choices / 対応するモデル選択

- `mode`: `single-csl` or `inter-csl`
- word rescaling / 単語確率補正:
  - `legacy_theta_sum` (default / 既定)
  - `empirical_unigram`
  - `none`
- theta posterior:
  - `legacy_importance` (default, 10,000 candidates / 既定1万候補)
  - `dirichlet`
- attribute sequence / 属性列:
  - `distinct`: simple MEC2, sampled without replacement / 単純MEC2・重複なし
  - `iid`: uniform independent assignments with replacement / 一様独立・重複可
- Inter-CSL interaction: `mh`, `no_communication`, or `all_accept`

The legacy rescaling is defined exactly as

\[
R_w = \sum_z \theta_{z,w},\qquad
\widetilde\theta_{z,w}
= \frac{\theta_{z,w}/R_w}{\sum_v \theta_{z,v}/R_v}.
\]

All word-probability consumers—word generation, category updates, attribute
updates, and Inter-CSL acceptance ratios—use the same transformed matrix.

legacy rescaling は上式のとおり全 `z` 行を単純加算します。単語生成、
カテゴリ更新、属性列更新、Inter-CSL の受理率は、すべて同じ変換済み行列を
参照します。

## Installation / インストール

Python 3.11 is supported.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Quick start / クイックスタート

Create a small deterministic fixture, then run either model with its example
configuration.

決定論的な小型 fixture を生成し、各モデルの設定例を使って実行します。

```bash
legacy-csl make-fixture fixture

# One-agent learning / 1エージェント学習
legacy-csl validate-data --config examples/single.yaml
legacy-csl learn --config examples/single.yaml
legacy-csl analyze --config examples/single.yaml

# Two-agent learning and prediction / 2エージェント学習・予測
legacy-csl validate-data --config examples/inter.yaml
legacy-csl learn --config examples/inter.yaml
legacy-csl predict --config examples/inter.yaml
legacy-csl analyze --config examples/inter.yaml
```

Example paths are resolved relative to the YAML file, not the current working
directory. The schema is intentionally strict and rejects unknown keys.

設定内の相対パスはカレントディレクトリではなく YAML ファイルを基準に
解決されます。設定スキーマは意図的に厳格で、未知のキーをエラーにします。

## Input data / 入力データ

The recommended compact representation is one `agentU.npz`, `agentA.npz`, or
`agentB.npz` file. Each archive contains 2-D float arrays `a`, `p`, `o`, `c`,
and may contain integer truth arrays `z_a`, `z_p`, `z_o`, `z_c`. Single-CSL
also uses a rectangular `single_words.json`.

The `U` in `agentU.npz` is retained only as a data-format identifier for
compatibility with existing processed datasets; the public model name is
Single-CSL.

The loader also accepts the processed legacy datadump layout under
`agentU/agentA/agentB/testssNNN`, including `object_center.txt`,
`target_object.txt`, `action.csv`, CNN-PCA object features, and RGB-BoF color
features. Raw images are neither required nor distributed.

推奨入力は `agentU.npz`、`agentA.npz`、`agentB.npz` です。各ファイルは
2次元 float 配列 `a/p/o/c` と、任意で正解 `z_a/z_p/z_o/z_c` を持ちます。
Single-CSL は長方形の `single_words.json` も使います。`agentU` の `U` は
既存データ形式との互換性のために残した識別子で、公開モデル名は
Single-CSL です。処理済み legacy datadump 形式も読み込めますが、画像本体は
不要です。

## Outputs / 出力

Each experiment writes:

- compressed native state checkpoints;
- resolved YAML;
- major compatibility CSVs (`z*`, `F`, `theta`, `Mu`, `Sig`, `pi`, words);
- metrics/history CSV and JSON;
- a learning-curve PNG and category-by-word heatmap PNG;
- Inter-CSL prediction arrays and MSE metrics when requested.

## Reproducibility / 再現性

Every stochastic operation receives an explicit NumPy generator derived from
`run.seed`. Repeating an experiment with the same input, configuration, and
dependency versions therefore follows the same random sequence.

全確率処理は `run.seed` から派生した明示的 NumPy generator を受け取ります。
同じ入力・設定・依存バージョンで再実行すれば、同じ乱数系列を使用します。

NPYLM, Pitman–Yor word generation, character rescaling, extended MEC2, deep
observation models, TensorFlow, and Torch are deliberately outside this
repository.

NPYLM、Pitman–Yor 単語生成、文字補正、extended MEC2、深層観測モデル、
TensorFlow、Torch は意図的に対象外です。

<details>
<summary>Repository creation note / レポジトリ作成者向けの補足</summary>

`legacy-csl` is a standalone, typed reorganization of the categorical
`uni-csl` and `inter-csl` execution paths formerly hosted under
`src/_legacy_codes` in `multi-agent-csl`.

`legacy-csl` は、`multi-agent-csl` の `src/_legacy_codes` にあった
categorical `uni-csl` / `inter-csl` の実行経路を、独立・型付きの構成へ
再整理したリポジトリです。旧クラス階層は引き継がず、設定、データ、状態、
確率計算、推論、保存、解析を分離しています。

The implementation was reorganized from the algorithms present at
`multi-agent-csl` commit `831140870e5d92adfd707c40597fc5d1409c1ec0`. The
Python 2 source under `lib/__original_source/rescaling_term.py` was used as the
numerical oracle for legacy theta rescaling and importance weights.

実装は上記コミット時点のアルゴリズムを基準に再整理しました。legacy theta
補正と importance 重みについては、Python 2 原版を数値 oracle として
テストしています。

</details>
