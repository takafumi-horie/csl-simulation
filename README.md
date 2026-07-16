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

`empirical_unigram` reproduces the legacy Inter-CSL comparison condition: the
ordinary Gibbs consumers use raw `theta`, while only the MH ratio divides the
old and proposed word probabilities by their respective corpus-snapshot
counts. It does not add smoothing or renormalize the rows. `none` uses raw
`theta` everywhere.

`empirical_unigram` は legacy Inter-CSL の比較条件を再現します。通常の Gibbs
更新では未補正の `theta` を使い、MH 受理率だけが旧単語・提案単語の確率を
それぞれのコーパス snapshot 頻度で割ります。加算 smoothing と行の再正規化は
行いません。`none` は全箇所で未補正の `theta` を使います。

## Faithful inference scope / 推論の再現範囲

The categorical learning kernel infers every latent variable updated by the
legacy Single-CSL and Inter-CSL paths:

- category assignments `z` for every observed object, not only the attended
  object;
- mixture proportions `pi` from all of those assignments;
- full-covariance Gaussian parameters `(mu, Sigma)` under the conjugate
  Gaussian--inverse-Wishart posterior;
- category-to-word distributions `theta`;
- one attribute sequence `F` per document; and
- in Inter-CSL, internal words through the token-wise MH naming game.

Words are observed in Single-CSL. Their lengths may differ by document, and
their vocabulary is encoded in first-seen order. The default legacy importance
kernel uses the scan `z -> pi -> (mu, Sigma) -> F -> theta`; the optional direct
Dirichlet kernel uses the later scan `z -> pi -> (mu, Sigma) -> theta -> F`.
Prediction is generative: it resamples speaker categories and words, listener
categories, and listener observations rather than reusing training `z` or `F`.

categorical 学習 kernel は、旧 Single-CSL / Inter-CSL の実行経路が更新していた
潜在変数をすべて推論します。

- 注視対象だけでなく全観測物体のカテゴリ割当て `z`
- 全物体の割当てから更新する混合比率 `pi`
- Gaussian--inverse-Wishart 共役事後に基づく完全共分散 Gaussian パラメータ
  `(mu, Sigma)`
- カテゴリから単語への分布 `theta`
- 文書ごとの属性列 `F`
- Inter-CSL では token 単位 MH naming game による内部単語

Single-CSL の単語は観測値です。文書ごとに発話長が異なってもよく、語彙は
初出順で符号化します。既定の legacy importance kernel は
`z -> pi -> (mu, Sigma) -> F -> theta`、任意の直接 Dirichlet kernel は後期実装の
`z -> pi -> (mu, Sigma) -> theta -> F` という更新順です。予測時には学習済みの
`z` や `F` を流用せず、話者カテゴリ・単語、聞き手カテゴリ・聞き手観測を
生成的に再サンプルします。

The implementation matches the probability distributions and transition
kernel, but is not intended to reproduce the old program's random stream
bit-for-bit: it uses an explicit modern NumPy generator and a reorganized draw
schedule.

確率分布と遷移 kernel は旧実装と等価ですが、旧プログラムと乱数列まで
bit 単位で一致させるものではありません。明示的な現行 NumPy generator と
再整理した乱数呼出しを使用します。

## Installation / インストール

Python 3.11 is supported.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Running the tests / テストの実行

After installing the development dependencies as shown above, run the complete
test suite from the repository root:

上記の開発用依存をインストールした後、リポジトリのルートで全テストを
実行します。

```bash
python -m pytest
```

To run one test module or display individual test names and additional detail:

特定のテストモジュールだけを実行する場合や、各テスト名を含む詳細を表示する
場合は次のようにします。

```bash
python -m pytest tests/test_rescaling.py
python -m pytest -v
```

## Preparing existing data / 既存データの準備

Before using the commands below with an existing dataset, copy an example YAML
and edit at least `mode`, `paths.data_root`, `paths.output_root`, and the
category counts. `paths.data_root` must point to the directory that directly
contains `agentU.npz`/`agentA.npz`/`agentB.npz`, or the corresponding
`agentU`/`agentA`/`agentB` directories in the paper-dataset layout. Paths in a
YAML file are resolved relative to that YAML file.

既存データで以下のコマンドを使う前に、設定例の YAML をコピーし、少なくとも
`mode`、`paths.data_root`、`paths.output_root`、カテゴリ数を変更してください。
`paths.data_root` は、`agentU.npz` / `agentA.npz` / `agentB.npz` を直接含む
ディレクトリ、または論文データ形式の `agentU` / `agentA` / `agentB`
ディレクトリを直接含む場所を指定します。YAML内の相対パスは、そのYAML
ファイルを基準に解決されます。

Single-CSL also needs the observed utterances. For compact or newly prepared
data, edit `single_words.json` under `paths.data_root`. The outer JSON array
contains one utterance per document in the same document order, and each inner
array contains that document's word tokens. Rows must be non-empty; different
documents may have different utterance lengths. For example:

Single-CSL では観測発話も必要です。compact形式または新たに用意したデータを
使う場合は、`paths.data_root` 直下の `single_words.json` を編集してください。
JSONの外側配列はデータと同じ文書順に1文書1発話を並べ、各内側配列にその
文書の単語tokenを記述します。空の発話は使えませんが、発話長は文書ごとに
異なっても構いません。例えば次の形式です。

```json
[
  ["push", "red", "left", "box"],
  ["touch", "blue", "right", "ball"]
]
```

For paper-dataset reproduction, the legacy
`agentU/testss(001-040)/testss_words.csv` is read automatically when neither
`single_words.json` nor `uni_words.json` exists. Adding `single_words.json`
overrides that CSV, which is useful when reproducing the observations with a
different utterance set. Inter-CSL generates its internal words and therefore
does not require this JSON file. Run `csl validate-data` after editing the YAML
and JSON, before starting a long inference run.

論文データの再現では、`single_words.json` と `uni_words.json` が無ければ legacy
の `agentU/testss(001-040)/testss_words.csv` を自動的に読み込みます。
`single_words.json` を置くとCSVより優先されるため、同じ観測に別の発話集合を
与える実験にも利用できます。Inter-CSL は内部単語を生成するため、このJSONは
不要です。長時間の推論を始める前に、YAMLとJSONの編集後に
`csl validate-data` を実行してください。

## Quick start / クイックスタート

Create a small deterministic fixture, then run either model with its example
configuration.

決定論的な小型 fixture を生成し、各モデルの設定例を使って実行します。

```bash
csl make-fixture fixture

# One-agent learning / 1エージェント学習
csl validate-data --config examples/single.yaml
csl learn --config examples/single.yaml
csl analyze --config examples/single.yaml

# Two-agent learning and prediction / 2エージェント学習・予測
csl validate-data --config examples/inter.yaml
csl learn --config examples/inter.yaml
csl predict --config examples/inter.yaml
csl analyze --config examples/inter.yaml
```

Example paths are resolved relative to the YAML file, not the current working
directory. The schema is intentionally strict and rejects unknown keys.

設定内の相対パスはカレントディレクトリではなく YAML ファイルを基準に
解決されます。設定スキーマは意図的に厳格で、未知のキーをエラーにします。

## GMM hyperparameters / GMM ハイパーパラメータ

Each sensory attribute has its own Gaussian mixture. The mixture uses a
Dirichlet prior for category probabilities and a
Gaussian--inverse-Wishart prior for full-covariance Gaussian parameters.
Configure the legacy defaults under `hyperparameters.gmm` as follows:

各感覚属性は個別の Gaussian mixture を持ちます。カテゴリ確率には Dirichlet
事前分布、完全共分散 Gaussian のパラメータには
Gaussian--inverse-Wishart 事前分布を使います。旧実装と同じ既定値は
`hyperparameters.gmm` で次のように指定できます。

```yaml
hyperparameters:
  gmm:
    mixture_concentration: {a: 1.0, p: 1.0, o: 1.0, c: 1.0}
    frozen_mixture_concentration: {a: 0.1, p: 0.1, o: 0.1, c: 0.1}
    mean_prior: {a: 0.0, p: 0.0, o: 0.0, c: 0.0}
    mean_precision: {a: 0.001, p: 0.001, o: 0.001, c: 0.001}
    covariance_scale: {a: 0.01, p: 0.01, o: 0.01, c: 0.01}
    covariance_df_offset: {a: 2.0, p: 2.0, o: 2.0, c: 2.0}
    kmeans_n_init: 10
```

The keys `a`, `p`, `o`, and `c` mean action/haptic, position, object, and
color. Scalar mean and covariance settings are expanded across every feature
dimension of that attribute.

`a`、`p`、`o`、`c` はそれぞれ動作・触覚、位置、物体、色を表します。平均と
共分散に対するスカラー設定は、その属性の全特徴次元へ展開されます。

- `mixture_concentration` is the symmetric Dirichlet concentration for the
  category proportions `pi`. Values below `1` favor using fewer categories;
  larger values favor more even category proportions.
- `frozen_mixture_concentration` is the legacy alternative used when
  `freeze.gmm: true`; GMM variables are then kept fixed after initialization.
- `mean_prior` is the prior center of every Gaussian mean. Set it near the
  expected center of a normalized feature space.
- `mean_precision` is the prior strength for the Gaussian mean. Increasing it
  pulls posterior means more strongly toward `mean_prior`; decreasing it lets
  observations dominate sooner.
- `covariance_scale` forms the inverse-Wishart scale matrix
  `covariance_scale * I`. Larger values permit larger covariance at a fixed
  degree of freedom.
- `covariance_df_offset` defines the inverse-Wishart degrees of freedom as
  `feature_dimension + covariance_df_offset`. Larger values concentrate and,
  at fixed scale, shrink the covariance prior; tune it together with
  `covariance_scale`.
- `kmeans_n_init` controls how many K-means initializations are tried before
  inference. Increasing it can make initialization more robust but takes
  longer; it does not change the Bayesian prior.

- `mixture_concentration` はカテゴリ比率 `pi` の対称 Dirichlet 集中度です。
  `1` 未満では少数カテゴリを使う解を好み、大きくするとカテゴリ比率をより
  均等にする方向へ働きます。
- `frozen_mixture_concentration` は `freeze.gmm: true` のときに使う legacy
  側の集中度です。この場合、初期化後の GMM 変数は固定されます。
- `mean_prior` は各 Gaussian 平均の事前中心です。正規化済み特徴量で期待する
  中心付近に設定します。
- `mean_precision` は平均事前の強さです。大きくすると事後平均を
  `mean_prior` へ強く寄せ、小さくすると観測の影響が早く強くなります。
- `covariance_scale` は inverse-Wishart のスケール行列
  `covariance_scale * I` を作ります。自由度が同じなら、大きい値ほど大きな
  共分散を許します。
- `covariance_df_offset` は自由度を
  `特徴次元 + covariance_df_offset` と定めます。大きくすると分布が集中し、
  スケール固定時には共分散事前を小さくするため、`covariance_scale` と組で
  調整します。
- `kmeans_n_init` は推論前に試す K-means 初期化回数です。増やすと初期値が
  安定しやすい一方で計算時間が増えます。Bayes事前自体は変わりません。

Start from the legacy defaults, then tune one attribute at a time. Feature
standardization should be fixed before tuning because the mean and covariance
priors are expressed in feature-space units. Changing `category_counts` has a
larger structural effect than changing these priors and should be evaluated
separately.

まず legacy 既定値から始め、属性ごとに調整してください。平均・共分散事前は
特徴空間の単位に依存するため、調整前に特徴量の標準化方法を固定します。
`category_counts` の変更はこれらの事前分布より構造的影響が大きいため、分けて
評価してください。

## Input data / 入力データ

The recommended compact representation is one `agentU.npz`, `agentA.npz`, or
`agentB.npz` file. In each archive, `a` has shape `(D, d_a)`, `num_objects` and
`attentions` have shape `(D,)`, and each of `p`, `o`, and `c` has
`sum(num_objects)` rows. Object arrays may instead use shape `(D, M, d)` when
all documents have the same object count. Optional truth must provide all of
`z_a`, `z_p`, `z_o`, and `z_c`; object truth follows the same flattened object
order. Single-CSL additionally reads `single_words.json`, whose rows may have
different lengths.

The `U` in `agentU.npz` is retained only as a data-format identifier for
compatibility with existing processed datasets; the public model name is
Single-CSL.

For reproducing the paper experiments, use the processed legacy datadump layout
under `agentU/agentA/agentB/testssNNN`. The loader reads
`object_center.txt`, `target_object.txt`, `action.csv`, CNN-PCA object features,
and RGB-BoF color features directly from this layout. Raw images are neither
required nor distributed.

推奨入力は `agentU.npz`、`agentA.npz`、`agentB.npz` です。`a` は
`(D, d_a)`、`num_objects` と `attentions` は `(D,)`、`p/o/c` はそれぞれ
`sum(num_objects)` 行にします。全文書の物体数が同じ場合、物体配列は
`(D, M, d)` でも構いません。正解を含める場合は `z_a/z_p/z_o/z_c` をすべて
指定し、物体属性の正解は観測と同じ平坦化順にします。Single-CSL の
`single_words.json` は文書ごとに異なる発話長を使用できます。`agentU` の `U`
は既存データ形式との互換性のために残した識別子で、公開モデル名は
Single-CSL です。

論文実験を再現する場合は、`agentU/agentA/agentB/testssNNN` 以下に配置された
処理済み legacy datadump 形式を使用してください。この形式から
`object_center.txt`、`target_object.txt`、`action.csv`、CNN-PCA物体特徴、
RGB-BoF色特徴を直接読み込みます。生画像は必要なく、本リポジトリにも含まれ
ません。

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
Python 2 source at history commit
`53ac2bd679c69b9cc7fa218ff4ef0d65c6b7bc23`, path
`lib/__original_source/rescaling_term.py`, was used as the numerical oracle for
legacy theta rescaling and importance weights.

実装は上記コミット時点のアルゴリズムを基準に再整理しました。legacy theta
補正と importance 重みについては、履歴 commit
`53ac2bd679c69b9cc7fa218ff4ef0d65c6b7bc23` の
`lib/__original_source/rescaling_term.py` にある Python 2 原版を数値 oracle
としてテストしています。

</details>
