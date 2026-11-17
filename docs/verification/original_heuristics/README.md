# 原典ヒューリスティクスの状態・乱数比較記録

通常のテストとは独立した、短い軌道の完全一致検証を保存する。
本番パッケージや通常のテストは、このディレクトリ、監査用adapter、実データ、
原典ソースのいずれにも依存しない。長時間のARI探索や論文値の再現認定ではない。

## 初回記録の由来

- 機能実装: `74f45ff568dfa16d9410a5558038ff194e66a895`
- 変更前の比較対象: `c076b8b9bb86d99d10b8a261fcf9911108f7b890`
- 外部監査ブランチ: `2c662d13e104b6e8a600d2765cc44a8747427d92`
- 外部adapter: `investigation/exploratory_phase2_driver.py` の `B2current_init`
- adapter SHA256: `0a0cddafcdcac09fe7ad78656a85c9dbf9a7c670c7180c37d49b0c343b940dba`

[defaults_result_74f45ff.json](results/defaults_result_74f45ff.json) と
[b2_result_74f45ff.json](results/b2_result_74f45ff.json) は、最初の実行で
`/tmp/original-heuristics-parity/defaults_result.json` と `b2/result.json` に
生成されたJSONを**byte単位で無改変**で保存したもの。
初回の一時ディレクトリ名 `new_omitted_v3` / `new_explicit_v3` も保持した。

| 保存JSON | SHA256 |
|---|---|
| `defaults_result_74f45ff.json` | `31eeea5db49acf1115eacdc5a7269a4c439a9d3534f6938e60fc1d149b4f506c` |
| `b2_result_74f45ff.json` | `803e41d528d8f980cc83110b006f4afa3182c141f63de867bedffa301fe157fc` |

両記録の `feature_model_sha256` は、**docstring変更前の74f45ff時点**の
`src/legacy_csl/model.py` の
`0103943005244f583b863be0387da629a31439194eea12f7496eca480c0f721e`。
現在のモデルのハッシュを意味しない。後続のdocstring変更だけでもソースSHAは
変わる。再実行結果は状態配列・metadata・RNGの一致と比較件数を確認し、
保存JSON全体やソースSHAの一致を要求しない。

## 比較する境界

`trajectory.py` はcompact fixtureを生成し、両モードそれぞれ次の4組合せを
初期化直後と4反復後まで、計5境界で保存する。

- importance sampler + `legacy_theta_sum`
- Dirichlet sampler + `legacy_theta_sum` / `none` / `empirical_unigram`

importanceと他のrescalingの組合せは不正設定なので実行しない。
各条件はseed21、各属性2カテゴリ、語彙8の明示的な小規模設定を用いる。
例示YAMLのハイパラ比較ではなく、ヒューリスティクス省略時・明示無効時の
既存挙動を検証する。Singleは観測語を固定し、Interは初回proposal cacheと
A→BのMH通信順を維持する。
`compare_defaults.py` は旧版と新版省略/明示無効のそれぞれについて、
60 agent-state対、1320配列対のdtype・shape・raw bytes、
60 metadata対、60 canonical RNG JSON対がすべて一致することを確認する。

`compare_b2.py` は現行前処理の実データで、監査B2と4機能有効の本番モデルを
試行番号0/2/10、両主体、初期化直後と4通信反復後まで比較する。
各試行とエージェントのRNGは共通seed0から `SeedSequence.spawn()` で生成する。
全660配列対と30 agent境界のmetadata/RNGが一致した。
学習に入る直前のRNGも確認し、初期化時に学習用平均化が混入していないことを
含めて検証する。adapterの初期化・遷移overrideを参照し、通常実行へ取り込まない。

## 環境

初回記録と以下の保存スクリプト再検証に使った数値環境は、Python **3.11.18**、
NumPy **2.4.6**、SciPy **1.17.1**、scikit-learn **1.9.0**。
元の静的検証環境は Ruff **0.15.21**（`python -m ruff --version` で確認）。
今回の3スクリプトには Ruff **0.15.21** と **0.17.5** の両方で
`check` を実行し、追加指摘がないことを確認した。
NumPy/SciPy等の版を変えた場合、同じ乱数seedでもbyte一致を保証しない。
BLAS等は以下のとおり単一threadに固定する。

## compact比較の再実行

このリポジトリのrootから実行する。Python実体とsite-packagesは実際の環境へ
置き換えてよいが、上記の版を揃える。出力は毎回新しいディレクトリを使う。
スクリプトは既存の出力先を上書きしない。

```bash
python311=/tmp/csl-audit-python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11
site_packages=/home/horie_takafumi/workspace/multi_agent_csl/legacy-csl/.venv/lib/python3.11/site-packages
repo="$PWD"
scripts="$repo/docs/verification/original_heuristics"
verify_run=$(mktemp -d)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$verify_run/mpl"
mkdir "$verify_run/base"
git archive c076b8b src | tar -x -C "$verify_run/base"

PYTHONPATH="$site_packages:$verify_run/base/src" "$python311" \
  "$scripts/trajectory.py" --output-root "$verify_run/old"
PYTHONPATH="$site_packages:$repo/src" "$python311" \
  "$scripts/trajectory.py" --output-root "$verify_run/new_omitted"
PYTHONPATH="$site_packages:$repo/src" "$python311" \
  "$scripts/trajectory.py" --output-root "$verify_run/new_explicit" --explicit
PYTHONPATH="$site_packages:$repo/src" "$python311" \
  "$scripts/compare_defaults.py" --baseline "$verify_run/old" \
  --new-omitted "$verify_run/new_omitted" \
  --new-explicit "$verify_run/new_explicit" \
  --model-source "$repo/src/legacy_csl/model.py" \
  --output "$verify_run/defaults_result.json"
```

初回状態を再現する場合、新版側のcheckoutに74f45ffを使う。
現行checkoutを使う場合は、その時点の本番モデルの再検証になる。

## B2比較の再実行

上の環境変数に加えて、外部監査checkout、元の実データ、原典ソースが必要。
リポジトリ単体には含めない。監査checkoutは上記2c662d1を保持したものを用い、
`investigation/` 内の依存adapter一式を残す。スクリプトはadapterのSHAを検査する。
実データ・原典も初回監査と同じ内容を使う。

```bash
audit_checkout=/path/to/legacy-csl-ari-reproduction
data_root=/path/to/datadump
original_root=/path/to/multi-agent-csl/lib/__original_source
PYTHONPATH="$site_packages:$repo/src" "$python311" \
  "$scripts/compare_b2.py" \
  --adapter-dir "$audit_checkout/investigation" \
  --data-root "$data_root" --original-root "$original_root" \
  --output-root "$verify_run/b2"
```

既定の `--trials 0 2 10 --steps 4` が初回と同じ比較境界。
監査checkoutへのbytecode生成を無効にし、入力は読取り専用として扱う。
各スクリプトの `--help` で引数を確認できる。

## 保存スクリプト自体の検証

3本すべての`--help`、compact比較3軌道＋比較、B2の試行0/2/10×4反復を
保存版スクリプトで再実行し、上記と同じ比較件数ですべてPASSした。
その再実行時のmodel SHAは
`8c72b4e6a82da43f8e168f2555ab92661bbbe644da45f8e0bea7b22fedd95ccc`。
初回記録は置き換えず、再実行結果は別出力先に生成した。
