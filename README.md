# legacy-csl

`legacy-csl` is a standalone, typed reorganization of the categorical
`uni-csl` and `inter-csl` execution paths formerly hosted under
`src/_legacy_codes` in `multi-agent-csl`.

`legacy-csl` は、`multi-agent-csl` の `src/_legacy_codes` にあった
categorical `uni-csl` / `inter-csl` の実行経路を、独立・型付きの構成へ
再整理したリポジトリです。旧クラス階層は引き継がず、設定、データ、状態、
確率計算、推論、保存、解析を分離しています。

## Supported model choices / 対応するモデル選択

- `mode`: `uni-csl` or `inter-csl`
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

Create the distributable synthetic fixture and use an example configuration.

```bash
legacy-csl make-fixture fixture
legacy-csl validate-data --config examples/uni.yaml
legacy-csl learn --config examples/uni.yaml
legacy-csl analyze --config examples/uni.yaml

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
and may contain integer truth arrays `z_a`, `z_p`, `z_o`, `z_c`. Uni-CSL also
uses a rectangular `uni_words.json`.

The loader also accepts the processed legacy datadump layout under
`agentU/agentA/agentB/testssNNN`, including `object_center.txt`,
`target_object.txt`, `action.csv`, CNN-PCA object features, and RGB-BoF color
features. Raw images are neither required nor distributed.

推奨入力は `agentU.npz`、`agentA.npz`、`agentB.npz` です。各ファイルは
2次元 float 配列 `a/p/o/c` と、任意で正解 `z_a/z_p/z_o/z_c` を持ちます。
Uni-CSL は長方形の `uni_words.json` も使います。処理済み legacy datadump
形式も読み込めますが、画像本体は不要です。

## Outputs / 出力

Each experiment writes:

- compressed native state checkpoints;
- resolved YAML;
- major compatibility CSVs (`z*`, `F`, `theta`, `Mu`, `Sig`, `pi`, words);
- metrics/history CSV and JSON;
- a learning-curve PNG and category-by-word heatmap PNG;
- Inter-CSL prediction arrays and MSE metrics when requested.

## Reproducibility and provenance / 再現性と由来

Every stochastic operation receives an explicit NumPy generator derived from
`run.seed`. The implementation was reorganized from the algorithms present at
`multi-agent-csl` commit `831140870e5d92adfd707c40597fc5d1409c1ec0` and uses
the Python 2 source under `lib/__original_source/rescaling_term.py` as the
oracle for legacy theta rescaling and importance weights.

全確率処理は `run.seed` から派生した明示的 NumPy generator を受け取ります。
legacy theta 補正と importance 重みは Python 2 原版を数値 oracle として
テストします。

NPYLM, Pitman–Yor word generation, character rescaling, extended MEC2, deep
observation models, TensorFlow, and Torch are deliberately outside this
repository.

NPYLM、Pitman–Yor 単語生成、文字補正、extended MEC2、深層観測モデル、
TensorFlow、Torch は意図的に対象外です。

