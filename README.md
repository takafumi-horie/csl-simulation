# csl-simulation

`csl-simulation` provides reproducible reference implementations of two
categorical cross-situational learning models. The `single-csl` mode runs
**CSL-PGM**, which learns a word–meaning system from one agent's multimodal
observations and fixed utterances. The `inter-csl` mode runs **Inter-CSL-PGM**,
which lets two agents learn through a Metropolis–Hastings naming game and can
predict one agent's observations from the other's words.

`csl-simulation` は、カテゴリカル交差状況学習モデルを再現可能な形で
実行するためのリファレンス実装です。`single-csl` mode は、1エージェントの
マルチモーダル観測と固定発話から単語と意味の対応を学習する **CSL-PGM**
を実行します。`inter-csl` mode は、2エージェント間の Metropolis–Hastings
naming game によって記号体系を学習し、相手の単語から観測を予測できる
**Inter-CSL-PGM** を実行します。

The package includes strict YAML configuration, explicit random seeds,
checkpointing, legacy-compatible CSV export, standard metrics, and compact
plots. It can read either the included synthetic-data format or the processed
legacy datadump layout.

本パッケージには、厳格な YAML 設定、明示的乱数シード、チェックポイント、
主要な legacy 互換 CSV 出力、標準評価指標、基本的な可視化が含まれます。
同梱の合成データ形式と、処理済み legacy datadump 形式の両方を読み込めます。

## Models and papers / モデルと論文

- **CSL-PGM** (`single-csl`): [Cross-Situational Learning with Bayesian
  Generative Models for Multimodal Category and Word Learning in
  Robots](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2017.00066/full)
  / 1エージェントの交差状況学習モデル。
- **Inter-CSL-PGM** (`inter-csl`): [Symbol Emergence as Interpersonal
  Cross-Situational Learning: The Emergence of Lexical Knowledge with
  Combinatoriality](https://arxiv.org/abs/2306.15837)
  / 複数エージェント間の交差状況学習モデル。

The strings `single-csl` and `inter-csl` are configuration mode identifiers;
the model names used in this README are **CSL-PGM** and **Inter-CSL-PGM**.

`single-csl` と `inter-csl` は設定上の mode 識別子です。本 README では、
正式なモデル名として **CSL-PGM** と **Inter-CSL-PGM** を使用します。

### MLA-style citations / MLA形式の引用

- Taniguchi, Akira, et al. “Cross-Situational Learning with Bayesian Generative
  Models for Multimodal Category and Word Learning in Robots.” *Frontiers in
  Neurorobotics*, vol. 11, 19 Dec. 2017, article 66,
  <https://doi.org/10.3389/fnbot.2017.00066>.
- Hagiwara, Yoshinobu, et al. “Symbol Emergence as Interpersonal
  Cross-Situational Learning: The Emergence of Lexical Knowledge with
  Combinatoriality.” *arXiv*, 27 June 2023, arXiv:2306.15837,
  <https://doi.org/10.48550/arXiv.2306.15837>.

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
- Inter-CSL-PGM interaction: `mh`, `no_communication`, or `all_accept`

The legacy rescaling is defined exactly as

\[
R_w = \sum_z \theta_{z,w},\qquad
\widetilde\theta_{z,w}
= \frac{\theta_{z,w}/R_w}{\sum_v \theta_{z,v}/R_v}.
\]

All word-probability consumers—word generation, category updates, attribute
updates, and Inter-CSL-PGM acceptance ratios—use the same transformed matrix.

legacy rescaling は上式のとおり全 `z` 行を単純加算します。単語生成、
カテゴリ更新、属性列更新、Inter-CSL-PGM の受理率は、すべて同じ変換済み行列を
参照します。

`empirical_unigram` reproduces the legacy Inter-CSL-PGM comparison condition: the
ordinary Gibbs consumers use raw `theta`, while only the MH ratio divides the
old and proposed word probabilities by their respective corpus-snapshot
counts. It does not add smoothing or renormalize the rows. `none` uses raw
`theta` everywhere.

`empirical_unigram` は legacy Inter-CSL-PGM の比較条件を再現します。通常の Gibbs
更新では未補正の `theta` を使い、MH 受理率だけが旧単語・提案単語の確率を
それぞれのコーパス snapshot 頻度で割ります。加算 smoothing と行の再正規化は
行いません。`none` は全箇所で未補正の `theta` を使います。

## Faithful inference scope / 推論の再現範囲

The categorical learning kernel infers every latent variable:

- category assignments `z` for every observed object, not only the attended
  object;
- mixture proportions `pi` from all of those assignments;
- full-covariance Gaussian parameters `(mu, Sigma)` under the conjugate
  Gaussian--inverse-Wishart posterior;
- category-to-word distributions `theta`;
- one attribute sequence `F` per document; and
- in Inter-CSL-PGM, internal words through the token-wise MH naming game.

Words are observed in CSL-PGM. Their lengths may differ by document, and
their vocabulary is encoded in first-seen order. The default legacy importance
kernel uses the scan `z -> pi -> (mu, Sigma) -> F -> theta`; the optional direct
Dirichlet kernel uses the later scan `z -> pi -> (mu, Sigma) -> theta -> F`.
Prediction is generative: it resamples speaker categories and words, listener
categories, and listener observations rather than reusing training `z` or `F`.

categorical 学習 kernel は、潜在変数をすべて推論します。

- 注視対象だけでなく全観測物体のカテゴリ割当て `z`
- 全物体の割当てから更新する混合比率 `pi`
- Gaussian--inverse-Wishart 共役事後に基づく完全共分散 Gaussian パラメータ
  `(mu, Sigma)`
- カテゴリから単語への分布 `theta`
- 文書ごとの属性列 `F`
- Inter-CSL-PGM では token 単位 MH naming game による内部単語

CSL-PGM の単語は観測値です。文書ごとに発話長が異なってもよく、語彙は
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

CSL-PGM also needs the observed utterances. For compact or newly prepared
data, edit `single_words.json` under `paths.data_root`. The outer JSON array
contains one utterance per document in the same document order, and each inner
array contains that document's word tokens. Rows must be non-empty; different
documents may have different utterance lengths. For example:

CSL-PGM では観測発話も必要です。compact形式または新たに用意したデータを
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
different utterance set. Inter-CSL-PGM generates its internal words and therefore
does not require this JSON file. Run `csl validate-data` after editing the YAML
and JSON, before starting a long inference run.

論文データの再現では、`single_words.json` と `uni_words.json` が無ければ legacy
の `agentU/testss(001-040)/testss_words.csv` を自動的に読み込みます。
`single_words.json` を置くとCSVより優先されるため、同じ観測に別の発話集合を
与える実験にも利用できます。Inter-CSL-PGM は内部単語を生成するため、このJSONは
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

During learning, `csl` displays an outer progress bar for completed trials and
an inner bar labeled with the current trial ID (for example, `exp000`) and its
inference iteration. 学習中は、全試行の進捗を示す外側のbarと、現在の試行ID
（例: `exp000`）および推論iterationを示す内側のbarを表示します。

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
order. CSL-PGM additionally reads `single_words.json`, whose rows may have
different lengths.

The `U` in `agentU.npz` is retained only as a data-format identifier for
compatibility with existing processed datasets; the public model name is
CSL-PGM.

For reproducing the paper experiments, use the processed legacy datadump layout
under `agentU/agentA/agentB/testssNNN`. The loader reads
`object_center.txt`, `target_object.txt`, `action.csv`, CNN-PCA object features,
and RGB-BoF color features directly from this layout. Raw images are neither
required nor distributed. Place each ground-truth directory inside its agent:

```text
<data_root>/
├── agentU/
│   └── testss(001-040)/
│       ├── za_C.csv
│       ├── zp_C.csv
│       ├── zo_C.csv
│       └── zc_C.csv
├── agentA/
│   └── testss(001-040)/
│       └── za_C.csv, zp_C.csv, zo_C.csv, zc_C.csv
└── agentB/
    └── testss(001-040)/
        └── za_C.csv, zp_C.csv, zo_C.csv, zc_C.csv
```

推奨入力は `agentU.npz`、`agentA.npz`、`agentB.npz` です。`a` は
`(D, d_a)`、`num_objects` と `attentions` は `(D,)`、`p/o/c` はそれぞれ
`sum(num_objects)` 行にします。全文書の物体数が同じ場合、物体配列は
`(D, M, d)` でも構いません。正解を含める場合は `z_a/z_p/z_o/z_c` をすべて
指定し、物体属性の正解は観測と同じ平坦化順にします。CSL-PGM の
`single_words.json` は文書ごとに異なる発話長を使用できます。`agentU` の `U`
は既存データ形式との互換性のために残した識別子で、公開モデル名は
CSL-PGM です。

論文実験を再現する場合は、`agentU/agentA/agentB/testssNNN` 以下に配置された
処理済み legacy datadump 形式を使用してください。この形式から
`object_center.txt`、`target_object.txt`、`action.csv`、CNN-PCA物体特徴、
RGB-BoF色特徴を直接読み込みます。生画像は必要なく、本リポジトリにも含まれ
ません。ground truthディレクトリ `testss(001-040)` は、agentごとに
`agentU`、`agentA`、`agentB` の内側へ配置してください。

For a legacy datadump, `csl validate-data` requires this per-agent directory
and all four truth CSVs for every agent used by the selected mode. A missing or
misplaced file is reported before inference starts. Compact NPZ inputs may
still omit ground truth unless `freeze.gmm: true` is selected.

legacy datadumpでは、`csl validate-data` が選択modeで使う各agentについて、この
agent別ディレクトリと4つの正解CSVを必須検証します。不足または配置違いは推論
開始前にエラーとして表示します。compact NPZ入力では、`freeze.gmm: true` を
指定しない限りground truthは引き続き省略できます。

## Outputs / 出力

Each experiment writes:

- compressed native state checkpoints;
- resolved YAML;
- major compatibility CSVs (`z*`, `F`, `theta`, `Mu`, `Sig`, `pi`, words);
- a complete compatibility-CSV snapshot after every inference iteration under
  `iterations/iterNNNN/csv/`;
- metrics/history CSV and JSON;
- a learning-curve PNG and category-by-word heatmap PNG (one heatmap for each
  agent in Inter-CSL-PGM); the heatmap removes
  categories never used by an attended object, labels rows with zero-based
  local IDs such as `p4`, labels columns with vocabulary strings, and preserves
  `a/p/o/c` blocks while arranging strong category-word pairs diagonally. For
  Inter-CSL-PGM, agent A determines the shared word order and attribute
  boundaries; agent B keeps that x-axis fixed and reorders rows only within
  each attribute block;
- ground-truth-category-by-word co-occurrence heatmaps for agents A and B in
  Inter-CSL-PGM when ground truth is available;
- Inter-CSL-PGM prediction arrays and MSE metrics when requested.

The iteration folders are one-based: `iter0001` is the state after the first
completed inference iteration and matches iteration 1 in `analysis/history.csv`.
Single-CSL writes all 21 state CSVs for agent U in each folder. Inter-CSL-PGM
writes all 42 state CSVs for agents A and B together, after both agents have
completed that outer iteration. The existing top-level `expNNN/csv/` remains
the final-state export for compatibility. Iteration folders contain CSV
snapshots only, not resumable native checkpoints. Because this creates 21 files
per agent per iteration, long runs can use substantial storage and file counts.

```text
exp000/
├── csv/                         # final-state compatibility CSVs
├── iterations/
│   ├── iter0001/csv/            # after iteration 1
│   ├── iter0002/csv/            # after iteration 2
│   └── ...
└── analysis/
```

各反復の完了後、状態のlegacy互換CSV一式を
`iterations/iterNNNN/csv/` に保存します。番号は1始まりで、
`iter0001` は `analysis/history.csv` のiteration 1と同じ時点です。
Single-CSLはUの21ファイル、Inter-CSL-PGMはA・B両方の更新が
完了した後に合計42ファイルを同じ反復フォルダに保存します。
従来の `expNNN/csv/` は最終状態の互換出力として維持します。
反復フォルダは解析用CSV snapshotであり、厳密な学習再開用checkpoint
ではありません。反復数が多い場合はストレージ使用量とファイル数に
注意してください。

カテゴリ×単語heatmapでは、注視対象に一度も使われなかったカテゴリを除外し、
行を `p4` のような0始まりの属性内カテゴリID、列を語彙文字列で表示します。
強いカテゴリ・単語対応が対角線状になるよう整列しますが、カテゴリ行は
`a/p/o/c` の属性blockを跨いで移動しません。Inter-CSL-PGMではエージェントA・Bを
`category_word_heatmap_A.png` と `category_word_heatmap_B.png` に分けて出力します。
両図の比較を容易にするため、Aを基準に決めた単語列と属性境界をBでも
共有し、Bはその固定列に対して属性block内の行だけを並べ替えます。

When Inter-CSL-PGM ground truth is available, the analysis also writes
`ground_truth_category_word_heatmap_A.png` and
`ground_truth_category_word_heatmap_B.png`. Each accepted word token is paired
with the attended ground-truth category of the attribute selected by its
inferred `F`; it is not paired with every category in the document. Object
truth is clipped with each agent's own attention. The plots show raw
co-occurrence counts, use labels such as `p4`, and share row and word orders
determined from the two agents' summed counts. Thus the category IDs are ground
truth, while the word-to-attribute correspondence still comes from inferred
`F`. A plot is omitted only for an agent whose dataset has no ground truth.

Inter-CSL-PGMでground truthが利用できる場合は、追加で
`ground_truth_category_word_heatmap_A.png` と
`ground_truth_category_word_heatmap_B.png` を出力します。これらの図は、
各受理済み単語を推論済み属性列 `F` が指す属性に割り当て、その
属性の注視対象ground-truthカテゴリとの生の共起回数を表示します。
`p/o/c` の正解はagentごとのattentionで注視物体へ切り詰めます。
行は `p4` のように属性と正解カテゴリIDを表示し、A/Bの行と列は
共起数を合算して決めた共通順に揃えます。したがって、カテゴリは
ground truthですが、単語の属性対応には推論結果 `F` が含まれます。
ground truthがないagentの図は出力しません。

### ARI and ground truth / ARIとground truth

ARI is computed with `sklearn.metrics.adjusted_rand_score`, so inferred
category numbers do not need to match ground-truth numbers; only the induced
partition matters. For `p`, `o`, and `c`, both inferred and true categories are
first clipped to each document's attended object. Action `a` already has one
category per document.

ARIは `sklearn.metrics.adjusted_rand_score` で計算するため、推論カテゴリ番号と
正解番号が直接一致する必要はなく、文書の分割が一致するかを評価します。
`p/o/c` では推論値と正解値の双方を各文書の注視対象物体へ切り詰めてから
計算します。Action `a` はもともと1文書1カテゴリです。

For CSL-PGM, `ari_truth_a/p/o/c` compares agent U with its ground truth. For
Inter-CSL-PGM, `ari_agents_a/p/o/c` compares agents A and B after selecting the
attended object with each agent's own attention indexes. If ground truth is
available, `ari_truth_A_a/p/o/c` and `ari_truth_B_a/p/o/c` are also written.
The older `ari_truth_a/p/o/c` names remain aliases for agent A for output
compatibility. The principal legacy Inter-CSL-PGM metric was the A-versus-B ARI;
ground-truth metrics are omitted when truth files are unavailable.

CSL-PGMの `ari_truth_a/p/o/c` はagent Uとground truthを比較します。
Inter-CSL-PGMの `ari_agents_a/p/o/c` は、A・Bそれぞれ自身のattention indexで
注視対象を選んだ後、両agentを比較します。ground truthがある場合は
`ari_truth_A_a/p/o/c` と `ari_truth_B_a/p/o/c` も出力します。従来の
`ari_truth_a/p/o/c` は出力互換性のためagent Aの別名として残します。旧
Inter-CSL-PGMの主要指標はA対BのARIであり、正解ファイルが無い場合はground-truth
ARIを出力しません。

Compact NPZ truth is read from the complete set `z_a`, `z_p`, `z_o`, and
`z_c`. The generated mock fixture also writes the four legacy CSV files under
each `<data_root>/agentX/testss(001-040)` directory. In the paper-dataset
layout, each agent reads truth only from its own directory and only when all
four files `za_C.csv`, `zp_C.csv`, `zo_C.csv`, and `zc_C.csv` exist.
Object-category rows must follow the same object order as the observations.

compact NPZでは `z_a/z_p/z_o/z_c` の4項目すべてをground truthとして読みます。
mock fixture生成時には、legacy CSV 4ファイルもagentごとの
`<data_root>/agentX/testss(001-040)` に出力します。論文データ形式でも各agentは
自身のディレクトリだけを参照し、`za_C.csv/zp_C.csv/zo_C.csv/zc_C.csv` が
すべて存在する場合だけ正解を読み込みます。物体カテゴリの行は観測と同じ
物体順にします。

## Reproducibility / 再現性

Every stochastic operation receives an explicit NumPy generator derived from
`run.seed`. Repeating an experiment with the same input, configuration, and
dependency versions therefore follows the same random sequence.

全確率処理は `run.seed` から派生した明示的 NumPy generator を受け取ります。
同じ入力・設定・依存バージョンで再実行すれば、同じ乱数系列を使用します。
