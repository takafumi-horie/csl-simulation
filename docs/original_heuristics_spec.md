# 原典の4ヒューリスティクスを選択する仕様

## 目的と対象

現行実装へ、調査でB2と呼んだ構成の4変更だけを選択式で追加する。
調査用ワークツリー `/tmp/legacy-csl-ari-reproduction` とその結果は保持する。
この実装は独立した `feature/original-heuristics` ブランチの
`/tmp/legacy-csl-original-heuristics` で行い、調査用スクリプトや結果を依存にしない。
仕様・実装・検証結果を同じブランチに保存し、ユーザーが指名するレビュワの
確認に供する。

「原典ヒューリスティクス版」は原典にある以下の学習処理を使う版を意味する。
原典の初期化・前処理・乱数生成器を含むプログラム全体や、論文値そのものを
再現するという意味ではない。100標本平均は単一標本と更新分布が異なる。

## 設定インターフェース

任意のトップレベルセクション `original_heuristics` を追加する。
省略時の値は以下のとおりで、既存の動作と乱数消費順を保つ。

```yaml
original_heuristics:
  theta_mean_samples: 1
  inference_theta: rescaled
  gaussian_mean_samples: 1
  empty_component_mean: prior
```

| キー | 許容値 | 動作 |
|---|---|---|
| `theta_mean_samples` | 正の整数 | 学習θの各事後Dirichlet行から実際にN標本を生成して算術平均する。1は従来の1標本。N>1は `word_model.theta_sampler: dirichlet` が必須で、importance方式との組合せは設定エラー |
| `inference_theta` | `rescaled` / `raw` | 学習時のzとFに渡すθ。`rescaled` は既存の `word_theta()`、`raw` は保存された補正前θを使う。語生成・MH受理・予測のθ選択は変えない |
| `gaussian_mean_samples` | 正の整数 | 占有カテゴリについて、Σを事後逆Wishartから1回生成し、そのΣを条件とするNormalから実際にN個のμを生成して平均する。1は従来のGIW更新。全4属性に適用し、空カテゴリには適用しない |
| `empty_component_mean` | `prior` / `original_uniform` | 空カテゴリのμの処理。`prior` は従来のGIW事前更新。`original_uniform` は位置の各次元を[-0.5,0.5)、他属性を[0,1)から独立一様生成し、その後にΣを従来の事前逆Wishartから1回生成する |

未知キー・未知の列挙値・真偽値や浮動小数点の標本数・0以下の標本数は
設定エラーにする。解決済みYAMLには省略された既定値も保存し、再読込できる。
`inference_theta: rescaled` は `word_model.rescaling` の現在の動作を尊重する。
`none` や `empirical_unigram` の通常の単語θは現行どおり補正前となる。

公開メソッド `z_logweights()` のθ省略時は従来どおり `word_theta()` を使う。
`inference_theta: raw` による学習時の切替は `step()` がθを明示的に渡して行う。

### 更新順序とB2との対応

原典θ平均化を有効にした `theta_mean_samples > 1` では、B2と同じ
`z → pi → Gaussian → F → theta` 順を使う。
この順序はθ平均化の原典更新の一部として固定し、5番目の独立設定は追加しない。
`theta_mean_samples: 1` では従来どおりimportanceはF→θ、Dirichletはθ→Fを保つ。
後者で `inference_theta: raw` を指定した場合、zには更新前、Fには更新後の
補正前θを渡す。

B2は次の設定に対応する。1標本への復帰や他の3機能の切替は独立に指定できる。

```yaml
word_model:
  rescaling: legacy_theta_sum
  theta_sampler: dirichlet
  importance_candidates: 10000
original_heuristics:
  theta_mean_samples: 100
  inference_theta: raw
  gaussian_mean_samples: 100
  empty_component_mean: original_uniform
```

## 維持する挙動

- 初期化は現行のまま。初期θは1標本、初期Σは既存設定のスケール。
  学習標本数はconstructorの初期θ・語・KMeans・z・pi・Fや乱数境界に影響しない。
- `freeze.gmm` による正解ラベルからの初期GMM構築も従来のGIW更新を使い、
  学習時のμ平均・空カテゴリ設定を初期化へ混入させない。
- `freeze.gmm/theta/attributes` の対象更新は従来どおり実行しない。
- 前処理、通信順序、提案cache、語生成、MH受理判定、予測、保存状態の形式は維持する。
- 全設定を省略した既存設定の初期状態・複数反復後の状態・RNGは旧版と一致する。

## 設定例とREADME

既存の `examples/single.yaml` は変更しない。SINGLEの論文実験に対応する
語彙14・各属性の集中度1.0を、通常版・追加版の両方で維持する。
既存の `examples/inter.yaml` は学習方式を維持し、ハイパラを
INTERの論文実験に対応する語彙13・各属性の集中度0.1へ変更する。
コメントも `exclusivity: distinct` に合わせて「重複なし」に修正する。
次の別ファイルを追加し、READMEで通常版と並べて案内する。

| 新しいファイル | 4機能 | ハイパラ |
|---|---|---|
| `examples/single_original_heuristics.yaml` | 上記B2の4機能を有効化 | 既存singleの値を維持（語彙14、`gmm.mixture_concentration` 全属性1.0） |
| `examples/inter_original_heuristics.yaml` | 上記B2の4機能を有効化 | 語彙13、`gmm.mixture_concentration` 全属性0.1。その他は既存interと同じ |

READMEでは、α/Kは `hyperparameters.gmm.mixture_concentration` の各属性値に
対応すること、既存設定との切替方法、原典ヒューリスティクスは厳密な単一標本
Gibbs更新とは異なることを説明する。全く同一の論文実装とは呼ばない。

## 検証と受入条件

1. 設定の省略・部分指定・不正値・組合せ制約・解決済みYAML往復を検証する。
2. 標本数1と無効設定が旧版の配列・RNGを保つことを確認する。
3. θとμの平均化は独立した直接計算とRNGまで照合し、分散を縮めた1標本で
   代用していないことを確認する。1次元と空カテゴリも扱う。
4. z/Fのraw/rescaled選択、各更新順序、freeze、語生成・MH・予測の維持を検証する。
5. B2有効設定が、保存された調査用構成と同じ入力・seedで初期状態および
   複数反復後の状態・RNGに一致することを確認する。調査コードは検証時だけ
   参照し、本番パッケージ・通常のテストの依存にしない。
6. 両モードの学習・保存・予測/解析の結合テストと既存テスト、静的チェックを
   実行する。長時間のARI探索はやり直さない。
7. 既存singleが変更されず、既存interの変更がハイパラと上記コメントだけであること、
   SINGLEの両例が語彙14・集中度1.0、INTERの両例が語彙13・集中度0.1であり、
   新2例だけが4機能を有効化していることを検証する。

## 今回含めないもの

原典初期化への切替、初期Σ/θの追加設定、属性別標本数、任意の一様範囲、
独立の更新順序設定、探索runner、ARI探索の実験結果、原典コードの同梱は追加しない。
整合性検証のスクリプトと比較結果は、通常のテストから独立した
`docs/verification/original_heuristics/` に保存する。

## 実施した検証

初回実装 `74f45ff` では次を検証した。

- Python 3.11で既存・追加テスト計164件が通過。設定の検証、直接標本生成と
  RNGの照合、1次元・空カテゴリ、更新順序、freeze、語生成・MH・予測、
  両モードの学習・保存・再読込・解析を含む。予測はINTERで検証する。
- Ruff 0.15.21で `python -m ruff check .`、および `git diff --check` が通過。
  レビュワによるRuff 0.17.5の指摘12件は変更前にも存在し、初回実装による
  増加はなかった。静的チェックの結果は実行したRuff版と併せて扱う。
- 変更前 `c076b8b` と、新実装の設定省略・明示的無効化を別プロセスで比較した。
  両モード、importance＋legacy補正、Dirichlet＋3補正方式の計8条件について、
  初期化と4反復後までの各境界を照合。各比較60状態・1320配列・60メタデータ・
  60 RNG状態が完全一致した。記録と再実行方法は
  [保存した検証資料](verification/original_heuristics/README.md)を参照。
- B2は保存済みの調査用構成と実データで比較した。試行番号0・2・10の
  両エージェントについて、初期化と4回のMH通信・更新の各境界を照合し、
  計30状態の660配列、メタデータ、RNG状態が完全一致した。
  対照adapterのSHA-256は
  `0a0cddafcdcac09fe7ad78656a85c9dbf9a7c670c7180c37d49b0c343b940dba`。
  各試行・エージェントのRNGは共通seed 0から `SeedSequence.spawn()` で生成した。
  比較結果も上記の検証資料に保存する。
- 既存 `single.yaml` は変更前とバイト単位で同一。
  初回実装時点の既存 `inter.yaml` の差分は語彙数と混合事前集中度のみ。
  新2例は対応する通常例と同じハイパラで、4機能を有効化している。

長期実験のARIを取り直す検証ではなく、既に評価した構成の処理を本番実装へ
移した際の整合性検証である。保存した検証スクリプト・調査コードは本番および
通常のテストの依存に含めていない。

### レビュー指摘への対応

- 比較結果JSONを初回実装時の記録として保存し、比較スクリプトに入力先を
  指定して再実行する手順を添えた。上記の検証資料を参照。
- importance＋平均化なし＋rawで、z/Fに更新前θを渡すことを追加検証した。
  GMMだけfreezeした4機能有効設定も追加し、初期状態の一致、GMMの固定、
  F/θの更新順、事後計算・最終RNGを独立した計算と照合した。
  追加後はPython 3.11で全167件が通過した。
- `z_logweights()` のθ省略時は `word_theta()` を使う旨をdocstringに明記した。
  計算処理は変更していない。
- `inter.yaml` のコメントを「重複なし」に修正し、仕様・READMEも整合させた。
- Ruff 0.17.5について、変更前 `c076b8b` と今回のsrc/testsの指摘が同じ12件
  であることも再確認した。
- 保存版スクリプトでも既定設定8条件とB2の短い軌道を再実行し、配列・
  メタデータ・RNGの一致を確認した。新スクリプトはRuff 0.15.21/0.17.5の
  両版で通過し、全体は0.15.21で通過、0.17.5の指摘は既存12件から増加なし。
