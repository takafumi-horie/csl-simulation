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

Inter-CSL-PGM prediction follows the conditional chain
`A observation -> A category -> A word -> B category -> B observation`.
For every action and object observed by A, it samples the speaker category
from

\[
p(z^A=k\mid x^A) \propto
\pi^A_k\,\mathcal{N}(x^A;\mu^A_k,\Sigma^A_k).
\]

The learned global parameters remain fixed. Canonical `F=(a,p,o,c)` then maps
A's attended categories to words; B infers categories from those words and
generates predicted observations. The training-time document-local `z` and
`F` are not reused.

Inter-CSL-PGMの予測は
`Aの観測 -> Aのカテゴリ -> Aの単語 -> Bのカテゴリ -> Bの観測`
という条件付き生成を行います。Aの行動・全物体の観測に対して上式から
カテゴリをサンプルし、学習済み大域パラメータは固定します。その後、
canonical `F=(a,p,o,c)` とattentionでAの単語を生成し、Bが単語から
カテゴリと観測を予測します。学習時の文書固有 `z` と `F` は再利用しません。

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
- original implementation / 原典実装:
  - `execution_profile: current` (default): optional learning heuristics / 既定：学習時の4機能を選択可能
  - `execution_profile: original`: original-compatible training, with the SINGLE provenance limits below / 原典互換学習。SINGLEの出自上の限定は後述
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

With the default `original_heuristics.inference_theta: rescaled`, word
generation, category updates, attribute updates, and Inter-CSL-PGM acceptance
ratios use this transformed matrix. Selecting `inference_theta: raw` changes
only the training category and attribute updates; word generation, MH
acceptance, and prediction retain their existing theta handling.

legacy rescaling は上式のとおり全 `z` 行を単純加算します。既定の
`original_heuristics.inference_theta: rescaled` では、単語生成・カテゴリ更新・
属性列更新・Inter-CSL-PGM の受理率がこの変換済み行列を参照します。
`inference_theta: raw` は学習時のカテゴリ・属性列更新だけを未補正θへ切り替え、
単語生成・MH受理・予測のθの扱いは変えません。

`empirical_unigram` reproduces the legacy Inter-CSL-PGM comparison condition: the
ordinary Gibbs consumers use raw `theta`, while only the MH ratio divides the
old and proposed word probabilities by their respective corpus-snapshot
counts. It does not add smoothing or renormalize the rows. `none` uses raw
`theta` everywhere.

`empirical_unigram` は legacy Inter-CSL-PGM の比較条件を再現します。通常の Gibbs
更新では未補正の `theta` を使い、MH 受理率だけが旧単語・提案単語の確率を
それぞれのコーパス snapshot 頻度で割ります。加算 smoothing と行の再正規化は
行いません。`none` は全箇所で未補正の `theta` を使います。

### Original implementation heuristics / 論文オリジナル実装ヒューリスティクス

The optional top-level `original_heuristics` section selects an execution
profile and four learning heuristics. Omitting `execution_profile` selects
`current`: the existing initialization, preprocessing, independent NumPy
Generators, and communication order. Omitting the whole section also disables
the four heuristics, as shown below. The ordinary examples keep this behavior.

任意のトップレベル `original_heuristics` で、実行プロファイルと4つの学習処理を
選択できます。`execution_profile` の省略時は `current` で、従来の初期化・前処理・
独立したNumPy Generator・通信順を使います。セクション全体を省略すると、
4機能も以下の既定値となります。通常版の設定例はこの動作を維持します。

```yaml
original_heuristics:
  execution_profile: current
  theta_mean_samples: 1
  inference_theta: rescaled
  gaussian_mean_samples: 1
  empty_component_mean: prior
```

| Option / 設定 | Default / 既定値 | Behavior / 動作 |
|---|---|---|
| `execution_profile` | `current` | `current` preserves the existing execution; `original` selects the complete training profile described below. / `current` は従来方式、`original` は後述の学習実行方式一式を選択します。 |
| `theta_mean_samples` | `1` | Positive integer. `1` preserves the configured theta sampler; N > 1 averages N actual posterior Dirichlet draws per row and requires `word_model.theta_sampler: dirichlet`. Importance sampling with averaging is rejected. / 正の整数。`1` は選択したθ抽選方式を維持し、2以上は各事後Dirichlet行から実際に指定数を生成して平均します。`dirichlet` が必須で、importance方式との併用はエラーです。 |
| `inference_theta` | `rescaled` | `rescaled` uses the ordinary `word_theta()`; `raw` passes stored, uncorrected theta only to training `z` and `F`. / `rescaled` は通常の `word_theta()`、`raw` は保存された補正前θを学習時の `z` と `F` だけに渡します。 |
| `gaussian_mean_samples` | `1` | Positive integer: for each occupied category, draw Sigma once, then average this many actual conditional Normal draws of mu. Applies to all four attributes, not empty categories. / 占有カテゴリでΣを1回生成後、そのΣを条件とするNormalから指定数のμを実際に生成して平均します。全4属性が対象で、空カテゴリには適用しません。 |
| `empty_component_mean` | `prior` | `prior` keeps the GIW prior update. `original_uniform` draws each mu coordinate uniformly from `[-0.5, 0.5)` for position and `[0, 1)` for other attributes, then draws Sigma once from the existing prior inverse-Wishart. / `prior` は従来のGIW事前更新。`original_uniform` は位置を `[-0.5, 0.5)`、他属性を `[0, 1)` から座標ごとに独立一様生成し、その後に従来の事前逆WishartからΣを1回生成します。 |

Under `execution_profile: current`, with `theta_mean_samples > 1`, the scan is coupled to theta averaging:
`z -> pi -> Gaussian -> F -> theta`. With `theta_mean_samples: 1`, the ordinary
orders remain: importance uses `F -> theta`, and direct Dirichlet uses
`theta -> F`. In the latter case, `inference_theta: raw` gives `z` the old raw
theta and `F` the newly sampled raw theta. The `rescaled` choice respects
`word_model.rescaling`; `none` and `empirical_unigram` retain their ordinary
raw `word_theta()` behavior.

`execution_profile: current` では、`theta_mean_samples > 1` のときθ平均化と組にして
`z -> pi -> Gaussian -> F -> theta` 順で更新します。`theta_mean_samples: 1`
なら従来どおりimportanceは `F -> theta`、直接Dirichletは `theta -> F` です。
後者で `inference_theta: raw` を指定すると、`z` は更新前、`F` は更新後の
補正前θを使います。`rescaled` は `word_model.rescaling` を尊重するため、
`none` と `empirical_unigram` の通常の `word_theta()` は従来どおり未補正です。

In the `current` profile, these four options change learning only. Initialization
still uses one prior theta
draw and the configured initial covariance scale; words, KMeans, `z`, `pi`,
`F`, and constructor RNG consumption are unchanged. GMM initialization from
truth under `freeze.gmm` also keeps the ordinary GIW update. Frozen variables
remain frozen. Preprocessing, communication, word generation, MH acceptance,
prediction, and checkpoint formats retain their ordinary behavior.

`current` で4機能を選んだ場合の変更対象は学習時だけです。
初期θは事前からの1標本、初期Σは既存設定のスケールで、
初期語・KMeans・`z`・`pi`・`F` と初期化の乱数消費は変えません。
`freeze.gmm` の正解ラベルによるGMM初期化も従来のGIW更新を使い、各freeze対象の
学習更新は実行しません。前処理・通信・語生成・MH受理・予測・保存形式も維持します。

Averaging is a different update distribution from a single-sample Gibbs draw,
not merely a more accurate calculation of that draw. For a fixed conditional
Normal distribution, averaging N independent mu samples divides its variance
by N; a mean of Dirichlet samples likewise differs from one Dirichlet draw.
Reproducing these heuristics and justifying an exact Gibbs sampler are separate
questions. See the [implementation specification](docs-for-developper/original_heuristics_spec.md).

標本平均は単一標本Gibbs更新とは分布が異なり、単に1標本の計算精度を高める処理では
ありません。条件付きNormal分布を固定したとき、N個の独立なμの平均の分散は
1/Nとなり、Dirichlet標本の平均も単一標本とは異なります。このヒューリスティクスの
挙動を再現することと、厳密なGibbs更新の正当性は区別します。
[実装仕様](docs-for-developper/original_heuristics_spec.md)も参照してください。

### Original execution profile / 原典実行プロファイル

`execution_profile: original` switches initialization and learning together:
uniform category assignments, the original attribute draws, 100-draw initial
theta means, shared-RNG KMeans (`n_init=10`), initial covariance `0.05 I`, and
GEM stick-breaking initial mixtures. Learning uses the original draw APIs and
arithmetic order, one pi draw, and 100-draw theta and occupied-Gaussian mean
averages. The GIW prior covariance scale and learning mixture concentrations
remain the configured values; they are separate from initialization constants.

`execution_profile: original` は初期化と学習をまとめて切り替えます。
一様カテゴリ割当て、原典の属性抽選、初期θの100標本平均、共有乱数を使う
KMeans（`n_init=10`）、初期共分散 `0.05 I`、GEM型の初期混合比率を使います。
学習も原典の抽選API・演算順に揃え、πは1標本、θと占有カテゴリのμは100標本を
平均します。GIW事前の共分散スケールと学習時の混合集中度には設定値を使い、
初期化の定数と区別します。

Each trial uses one local `numpy.random.RandomState(run.seed + experiment_id)`
continuously from initialization through learning; INTER shares it across A,
B, SciPy, and KMeans. The seed and every effective trial seed must be integers
in `[0, 2**32 - 1]`. INTER draws a fresh B-to-A proposal, updates A, then draws
a fresh updated-A-to-B proposal and updates B, without an unused final proposal.
Missing vocabulary IDs in INTER's initial words stop the trial with its seed;
the implementation does not redraw to fill them.

各試行は `numpy.random.RandomState(run.seed + experiment_id)` を1個作り、
初期化から学習終了まで連続して使います。INTERではA/B・SciPy・KMeansも共有します。
seedと全試行の実効seedは `[0, 2**32 - 1]` の整数に限定します。
INTERは新たなB→A提案、A更新、新たな更新済みA→B提案、B更新の順に進み、
最後に未使用の提案を生成しません。初期内部語に全語IDが現れなければseedを示して
停止し、不足語を埋めるための再抽選は行いません。

This profile requires **all** settings below, plus word length 4 and
`hyperparameters.gmm.kmeans_n_init: 10`. INTER requires `interaction: mh`.
Incompatible settings are errors, rather than being silently changed.
`current` continues to support the existing combinations, including freeze,
importance sampling, and other interaction and attribute modes.

このプロファイルでは以下の**全設定**に加え、語長4と
`hyperparameters.gmm.kmeans_n_init: 10` が必要です。INTERの通信は
`interaction: mh` に限定します。不整合は設定エラーとし、自動で値を変更しません。
`current` では従来のfreeze・importance方式・他の通信方式・属性抽選も選択できます。

```yaml
word_model:
  rescaling: legacy_theta_sum
  theta_sampler: dirichlet
original_heuristics:
  execution_profile: original
  theta_mean_samples: 100
  inference_theta: raw
  gaussian_mean_samples: 100
  empty_component_mean: original_uniform
attribute_model:
  exclusivity: distinct
freeze:
  gmm: false
  theta: false
  attributes: false
```

For INTER, this profile requires the original legacy datadump. It reproduces
the original reader's use of **B's minima and maxima for the first three action
columns of both A and B**. Compact `agentA.npz` / `agentB.npz` alone cannot
recover that preprocessing and are rejected. If both formats exist, `original`
selects legacy input; `current` keeps its NPZ priority. Data validation,
learning, analysis, and prediction use the same selected preprocessing. This
historical behavior is reproduced for comparison; the author regards the
current, agent-specific normalization as the more reasonable preprocessing.

INTERでは元のlegacy datadumpが必要です。原典readerの挙動に合わせ、
**A/B両方の動作先頭3列をBの最小値・最大値で正規化**します。
compact `agentA.npz` / `agentB.npz` だけでは復元できないためエラーとします。
両形式がある場合、`original` はlegacy入力、`current` は従来どおりNPZを優先します。
データ検証・学習・解析・予測で同じ前処理を使います。これは比較のための歴史的挙動で、
著者は現行の主体別正規化の方が合理的と考えています。

For SINGLE, the user relayed the author's assurance that INTER reused SINGLE
and should share its heuristics. We therefore apply the common initialization
and learning procedures verified in INTER to one agent U. **The dedicated
SINGLE Python 2 source is unavailable**, so this is not an independently
verified reproduction of the complete SINGLE original. U's observed words
remain fixed, vocabulary size 14 and per-category learning concentration 1.0
are retained in its example, and its existing preprocessing (including tactile
features and compact input support) remains in use. SINGLE preprocessing is
outside the original-reader equivalence guarantee; B-based normalization and
communication do not apply.

SINGLEについては、ユーザーを通じて「INTERはSINGLEの使い回しで、ヒューリスティクス等は
共通のはず」という著者の説明を得ています。これを根拠に、INTERで確認した共通の
初期化・学習処理をUへ適用します。**SINGLE専用のPython 2原典は未入手**なので、
SINGLE原典全体との独立照合済み実装とは位置付けません。Uの観測語を固定し、
設定例の語彙14・学習時のカテゴリ当たり集中度1.0と、触覚を含む既存の前処理・
compact入力対応を維持します。SINGLEの前処理は原典readerとの同一性の保証対象外で、
Bの値域による正規化や通信は導入しません。

The INTER compatibility target is the identified original translated under
fixed modern Python and dependency versions, checked at initialization and
intermediate state/RNG boundaries. It does not promise bitwise identity with
the historical Python 2 environment or the paper's ARI values. Category counts
and other hyperparameters remain configurable; accepting other values is not
paper-setting reproduction. The fixed reference uses a batched 100-sample
Normal draw, so bitwise identity with 100 separate Python 2 calls is also
outside scope. Non-finite probabilities or invalid posterior matrices stop
with context, without algorithm substitution or redraws.

INTERの互換対象は、同定した原典を固定した現代のPython・依存版へ移した参照実装で、
初期状態と途中の状態・RNG境界で照合します。当時のPython 2環境とのbit単位一致や
論文ARI値は保証しません。カテゴリ数等は設定可能ですが、他の値で動くことと論文設定の
再現は区別します。参照版はNormalの100標本を一括生成するため、Python 2原文の
100回の逐次呼出しとのbit単位一致も保証対象外です。非有限確率や不正な事後行列では
文脈を示して停止し、別アルゴリズムへの置換や再抽選は行いません。

Compatibility covers **training**, not the complete original prediction
program. INTER prediction retains the existing conditional algorithm and
prediction-seed policy while sharing training's preprocessing; SINGLE
prediction remains unsupported. Checkpoint arrays and legacy outputs keep
their existing formats. Resolved configuration and a separate provenance
record identify the profile, seeds, input hashes, dependency versions, numerical
thread environment and loaded BLAS/OpenMP runtimes, and SINGLE/INTER scope.
Initialization and every completed half-turn retain the complete MT19937 state
in losslessly compressed `rng_boundaries.jsonl.gz`; read it as gzip JSON Lines.
A 500-iteration INTER trial records 1,001 boundaries. These records do not
consume random draws or add training resume support. Production has
no runtime dependency on original source or audit checkouts. See the
[execution specification](docs-for-developper/original_compatibility_spec.md)
and [verification records](docs-for-developper/verification/original_compatibility/).

互換性は**学習実行**を対象とし、原典の予測プログラム全体には拡張しません。
INTERの予測は既存の条件付きアルゴリズムと予測seed規則を使い、前処理を学習と揃えます。
SINGLEの予測は非対応です。checkpoint配列・legacy出力の形式を維持し、解決済み設定と
別の出自記録へprofile・seed・入力ハッシュ・依存版・数値計算のスレッド環境と
読込済みBLAS/OpenMPランタイム・SINGLE/INTERの保証範囲を保存します。
初期化完了時と各半ターン完了時のMT19937状態全体は、無損失圧縮した
`rng_boundaries.jsonl.gz` にgzip形式のJSON Linesで保存します。
INTERの500反復では1,001境界を記録します。記録処理は乱数を消費しません。
学習再開機能の追加ではありません。本番は原典ソースや監査checkoutへ実行時に依存しません。
[実行仕様](docs-for-developper/original_compatibility_spec.md)と
[検証記録](docs-for-developper/verification/original_compatibility/)も参照してください。

## Faithful inference scope / 推論の再現範囲

The categorical learning kernel infers every latent variable:

- category assignments `z` for every observed object, not only the attended
  object;
- mixture proportions `pi` from all of those assignments;
- full-covariance Gaussian parameters `(mu, Sigma)` using the conjugate
  Gaussian--inverse-Wishart posterior, with optional mean averaging and empty
  component handling described above;
- category-to-word distributions `theta`;
- one attribute sequence `F` per document; and
- in Inter-CSL-PGM, internal words through the token-wise MH naming game.

Words are observed in CSL-PGM. Under `current`, their lengths may differ by document, and
their vocabulary is encoded in first-seen order. With `theta_mean_samples: 1`,
the default legacy importance kernel uses the scan
`z -> pi -> (mu, Sigma) -> F -> theta`; the optional direct Dirichlet kernel
uses `z -> pi -> (mu, Sigma) -> theta -> F`. Averaging theta over multiple
draws uses the former, `F -> theta`, order.
Prediction is generative: it resamples speaker categories and words, listener
categories, and listener observations rather than reusing training `z` or `F`.

categorical 学習 kernel は、潜在変数をすべて推論します。

- 注視対象だけでなく全観測物体のカテゴリ割当て `z`
- 全物体の割当てから更新する混合比率 `pi`
- Gaussian--inverse-Wishart 共役事後に基づく完全共分散 Gaussian パラメータ
  `(mu, Sigma)`（平均化・空カテゴリ処理は上記の設定で選択可能）
- カテゴリから単語への分布 `theta`
- 文書ごとの属性列 `F`
- Inter-CSL-PGM では token 単位 MH naming game による内部単語

CSL-PGM の単語は観測値です。`current` では文書ごとに発話長が異なってもよく、語彙は
初出順で符号化します。`theta_mean_samples: 1` では、既定の legacy importance kernel は
`z -> pi -> (mu, Sigma) -> F -> theta`、任意の直接 Dirichlet kernel は後期実装の
`z -> pi -> (mu, Sigma) -> theta -> F` という更新順です。θを複数標本で平均する場合は
前者の `F -> theta` 順を使います。予測時には学習済みの
`z` や `F` を流用せず、話者カテゴリ・単語、聞き手カテゴリ・聞き手観測を
生成的に再サンプルします。

The selected profile and heuristics define the learning kernel. Exact
trajectory comparisons require the same input, settings, and numerical
environment; the original profile's narrower provenance and prediction limits
are described above.

実行プロファイルとヒューリスティクスの設定が学習kernelを定めます。
軌道の完全一致には同じ入力・設定・数値環境が必要です。原典プロファイルの
出自・予測に関する保証範囲は前述のとおりです。

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

Choose the ordinary or original-profile example for either model. Both
CSL-PGM examples keep vocabulary size 14 and per-category mixture concentration
1.0; both Inter-CSL-PGM examples use vocabulary size 13 and concentration 0.1.
The ordinary `single.yaml` and `inter.yaml` are unchanged. The two
`*_original_heuristics.yaml` files now select `execution_profile: original`,
changing their initialization, RNG, and (for INTER) preprocessing and
communication as well as enabling all four heuristics. INTER's original
example needs the legacy datadump, rather than the compact quick-start fixture.

各モデルで通常版と原典プロファイル版を選べます。SINGLEの両設定は語彙14・
カテゴリ当たり混合集中度1.0、INTERの両設定は語彙13・集中度0.1を維持します。
通常の `single.yaml` と `inter.yaml` は変更しません。
今回から2つの `*_original_heuristics.yaml` は `execution_profile: original` を選び、
4機能に加え初期化・乱数と、INTERの前処理・通信も変わります。
INTERの原典設定にはcompactのクイックスタートfixtureではなくlegacy datadumpが必要です。

To recover the previous four-heuristic behavior, set `execution_profile:
current` in either original example (or omit that key), leaving the four
heuristic settings intact. Earlier experiment reports remain results for
their recorded commits and configurations, not results of the new profile.
The [earlier specification](docs-for-developper/original_heuristics_spec.md)
describes the four heuristics under `current`.

以前の4機能だけの動作へ戻す場合は、原典設定例の `execution_profile` を `current` にするか
そのキーを省略し、他の4キーは維持してください。以前の実験報告は記載されたコミット・
設定での結果として保持し、新しいプロファイルの結果とは読み替えません。
[従来仕様](docs-for-developper/original_heuristics_spec.md)は `current` での4機能の説明です。

| Model / モデル | Ordinary / 通常版 | Original profile / 原典プロファイル版 |
|---|---|---|
| CSL-PGM | [single.yaml](examples/single.yaml) | [single_original_heuristics.yaml](examples/single_original_heuristics.yaml) |
| Inter-CSL-PGM | [inter.yaml](examples/inter.yaml) | [inter_original_heuristics.yaml](examples/inter_original_heuristics.yaml) |

```bash
# Original execution profile / 原典実行プロファイル
csl validate-data --config examples/single_original_heuristics.yaml
csl learn --config examples/single_original_heuristics.yaml
csl analyze --config examples/single_original_heuristics.yaml

csl validate-data --config examples/inter_original_heuristics.yaml
csl learn --config examples/inter_original_heuristics.yaml
csl predict --config examples/inter_original_heuristics.yaml
csl analyze --config examples/inter_original_heuristics.yaml
```

Both original-profile examples include:

両原典プロファイル版の設定は次を含みます。

```yaml
word_model:
  rescaling: legacy_theta_sum
  theta_sampler: dirichlet
original_heuristics:
  execution_profile: original
  theta_mean_samples: 100
  inference_theta: raw
  gaussian_mean_samples: 100
  empty_component_mean: original_uniform
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
The Inter-CSL-PGM examples explicitly use the following hyperparameters.
The CSL-PGM examples retain vocabulary size 14 and `mixture_concentration: 1.0`
for every attribute; the other GMM settings below are shared:

各感覚属性は個別の Gaussian mixture を持ちます。カテゴリ確率には Dirichlet
事前分布、完全共分散 Gaussian のパラメータには
Gaussian--inverse-Wishart 事前分布を使います。INTERの設定例では次の値を
明示的に指定します。SINGLEの設定例は語彙数14・全属性の
`mixture_concentration: 1.0` を維持し、以下の他のGMM設定は共通です。

```yaml
hyperparameters:
  vocabulary_size: 13
  gmm:
    mixture_concentration: {a: 0.1, p: 0.1, o: 0.1, c: 0.1}
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

The original Inter-CSL-PGM nonparametric setting uses counts plus `alpha / K` in the pi
posterior. Each YAML `mixture_concentration` value is already this per-category
`alpha / K`; it is not divided by `category_counts` again. Thus Inter-CSL-PGM
`alpha=1, K=10` corresponds to `0.1` in each attribute. These example values
are explicit: omitting `mixture_concentration` still uses the schema default
`1.0`, and `original_heuristics` does not change hyperparameters automatically.

原典INTERのnonparametric設定では、piの事後分布にカウントと `alpha / K` を加えます。
YAMLの各属性の `mixture_concentration` はこのカテゴリ当たりの `alpha / K`
そのもので、`category_counts` で再度割りません。原典INTERの `alpha=1, K=10` は
各属性 `0.1` に対応します。設定例では明示的に指定しており、キー省略時の
スキーマ既定値は従来の `1.0` です。`original_heuristics` がハイパーパラメータを
自動変更することもありません。

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

Start from the example settings, then tune one attribute at a time. Feature
standardization should be fixed before tuning because the mean and covariance
priors are expressed in feature-space units. Changing `category_counts` has a
larger structural effect than changing these priors and should be evaluated
separately.

まず設定例の値から始め、属性ごとに調整してください。平均・共分散事前は
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
required nor distributed. Following the CSL-PGM paper and its original
preprocessing, each vector-quantized RGB histogram is L1-normalized so that its
elements sum to one; an all-zero histogram is left unchanged. Place each
ground-truth directory inside its agent:

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
ません。CSL-PGM論文と元実装の前処理に従い、ベクトル量子化済みRGB
ヒストグラムは要素の総和が1になるようL1正規化します（全要素が0なら変更
しません）。ground truthディレクトリ `testss(001-040)` は、agentごとに
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
- Inter-CSL-PGM prediction arrays, attended speaker categories
  (`speaker_z_a/p/o/c`), and MSE metrics when requested.

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

The `current` profile derives independent NumPy Generators from `run.seed`;
`original` uses one local RandomState per trial with seed
`run.seed + experiment_id`. Neither changes global NumPy or Python RNG state.
Repeating an experiment requires the same input, configuration, dependency
versions, numerical runtimes, and numerical thread environment to follow the
same trajectory. The independent parity checks use one numerical thread.
Production records the active controls (`OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS`) and actual loaded runtime
thread counts without changing them. Explicit seeds are a verification
convention; the original OS-derived seeds and paper random streams are unknown.

`current` は `run.seed` から独立したNumPy Generatorを派生し、`original` は
試行ごとに `run.seed + experiment_id` のローカルRandomStateを1個使います。
グローバルなNumPy・Pythonの乱数状態は変更しません。同じ軌道を再現するには、
入力・設定・依存版に加え、数値計算ランタイムとスレッド環境も揃える必要があります。
独立照合は数値計算のスレッド数を1に固定しています。本番は上記6個の環境変数と
実際の読込済みランタイムのスレッド数を記録し、値を変更しません。
明示seedは今回の検証規約で、原典のOS由来seedや論文実験の乱数列を復元したものではありません。
