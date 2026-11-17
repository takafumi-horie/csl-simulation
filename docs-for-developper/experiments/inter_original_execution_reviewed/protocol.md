# 原典実行プロファイルの10試行・500反復比較

## 固定条件

対象は承認済み仕様9.4のINTERのみ。SINGLE原典やSINGLE論文値の再現認定には用いない。
`examples/inter_original_heuristics.yaml` の `execution_profile: original` を使い、
変更するのは絶対入力/出力パス、出力名、試行数10のみ。反復数500、base seed0、
語彙13、各属性10カテゴリ、γ0.1、学習πのカテゴリ当たり集中度0.1を維持する。
各試行の実効seedは0〜9、共有ローカルRandomStateを初期化から500反復まで連続消費する。

学習は本番 `runner._learn_original` を試行単位で呼び、`learn()` と同じconfig、
seed、experiment_idを渡す。各workerは初期化前に本番 `validate_data()` を実行する。
本番の更新・通信・CSV保存・checkpoint・解析・RNG境界記録を変更しない。
原典ソースや調査adapterを学習へimportしない。supervisorが通常の全試行設定を
`resolved_config.yaml` として保存する。

全試行を500反復まで実行し、結果による条件変更・試行選別・最良反復選択・早期終了は
行わない。失敗時は新規試行の起動を止め、起動済み試行の結果を保存し、失敗を記録する。
再抽選・自動再開は行わない。全完了が確認できるまで集計を合格扱いにしない。

## 実行と資源

BLAS/OpenMP/MKLをプロセスごとに1 threadへ固定する。最大並列数は1〜10で指定可能で、
試行のseedや遷移は変わらない。準備時にはCPU affinity24、親cgroupの
`cpu.max: max 100000`、メモリ制限maxを確認したため、既定値を10並列とした。
実行時のaffinity・cgroup制限・メモリ・空き容量もmanifestへ保存する。

開始にはrootの明示指示を待つ。準備だけを行う場合は以下を使う。
出力パスは環境に合わせて指定し、既存の本番出力や完成済みartifactの上では再実行しない。

```bash
repo=/tmp/legacy-csl-original-compatibility
artifact="$repo/docs-for-developper/experiments/inter_original_execution_reviewed"
python311=/tmp/csl-audit-python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11
site_packages=/home/horie_takafumi/workspace/multi_agent_csl/legacy-csl/.venv/lib/python3.11/site-packages
data_root=/absolute/path/to/datadump
export PYTHONPATH="$site_packages:$repo/src"
export MPLCONFIGDIR="$repo/outputs/inter-original-mpl"
"$python311" "$artifact/run_experiment.py" --data-root "$data_root" --prepare-only
```

開始承認後、同じコマンドから `--prepare-only` を外し、必要なら
`--max-parallel 10` を付ける。supervisorは完了まで動かしておく。
ホスト停止や実行基盤による強制終了に対する永続性は保証しない。

HEADの固定ガードは置かず、開始時HEADと実際に使った全本番Python・設定例・pyproject・
実行/集計スクリプト・本プロトコルのSHA256を保存する。各worker開始/終了時に照合し、
コードや有効設定が変われば停止する。全入力ファイルのSHA256、処理済み観測/正解/注視/
物体数のdtype・shape・hash、数値ライブラリ版、thread環境、各workerのcommand・log・
開始/終了時刻・終了code・最終出力hashも保存する。最後に入力とソースの不変性を再確認する。

大量の本番出力はgitignore対象の
`outputs/inter_original_execution_reviewed/exp000`〜`exp009` に保持する。
軽量なconfig・manifest・trial_status・集計結果を本ディレクトリへ保存する。

## 比較・独立集計

500反復後のnative NPZを直接読み、正解ラベルとのARIを
`sklearn.metrics.adjusted_rand_score` で再計算する。本番analysisのARI関数は使わない。
主対象は動作30場面、位置/形状/色の注意対象30物体で、A/Bを別々に示す。
補助としてp/o/c全55物体と、試行内A/B単純平均も計算する。
A/B平均を20個の独立標本とせず、10試行の平均・標本標準偏差（ddof=1）・全試行値を示す。
native metricsとhistory最終行、checkpointと最終z CSVも独立値と照合する。
history500行と共有RNG1001境界の順序を確認するが、全500反復の中間状態を原典と
照合したという意味ではない。

比較列は以下の3条件と論文値を分ける。

- 新しい本番 `original`：今回新規の10試行・500反復。
- 原典参照：固定した現代化原典の既存mhr試行0〜9。参照0〜2は初期監査からの再利用、
  3〜9は追加試行であり、盲検の独立確認とは呼ばない。
- 以前の4機能版：コミット `12ef90f` の本番10試行。通常の前処理・初期化・通信と
  `SeedSequence(0).spawn(10)` → A/Bへspawnを使うため、今回と同じtrial番号でも
  対応する乱数軌道ではない。過去のB2確認10〜19とも混同しない。
- 論文Table3：既存 `paper_reference.json` の動作0.309、位置0.794、形状0.286、色0.949。
  取得済み出典と内容をそのまま用い、論文表にない標準偏差は補わない。

原典参照の `mhr_10_artifacts.tar.gz` 内に全20最終checkpointが保存されているため、
ARIだけでなく全状態を照合する。数値配列はdtype・shape・raw bytesの完全一致を要求する。
参照の `words(30,4)` はnativeの `words_flat` をreshapeし、attributesはmetadataから
`<U1` 配列へ、整数語彙はcanonicalな文字列IDをint64配列へ写して比較する。
それ以外のθ・z・π・μ・Σは同名キーを直接比較する。offsetは場面/物体数と照合する。
20項目×A/B×10試行の400配列対と、参照の最終ARI全160値の完全一致が受入条件である。
参照アーカイブを直接readし、既存監査checkoutの展開・変更は行わない。

```bash
audit=/absolute/path/to/legacy-csl-ari-reproduction/investigation
"$python311" "$artifact/aggregate_results.py" \
  --reference-archive "$audit/mhr_10_artifacts.tar.gz" \
  --reference-summary "$audit/mhr_10_summary.json"
```

集計は学習を実行せず、完成済み結果を上書きしない。再集計には新しい
`--destination /path/to/new-summary` を指定できる。比較元の全hashと出自を結果JSONへ記す。
論文の正解データはユーザー確認済みだが、集計母集団（全物体か注意対象か）とA/Bの
集約方法は未確定のままとする。論文値との差を同一条件の性能差や統計的同等性と断定しない。
完全一致は固定現代環境の原典参照に対するもので、当時のPython 2のbit一致や
論文値の完全再現へ拡張しない。
