# 本番runnerによるINTER原典ヒューリスティクス実験

`config.yaml` はコミット `12ef90fc2fc0ffec402087014ba85d1b09359fc5` の
`examples/inter_original_heuristics.yaml` を基に、実データ/出力の絶対パスと
`run.experiments: 10`、出力名だけを解決した設定である。
500反復、seed0の `SeedSequence(0).spawn(10)` に対応するtrial0–9をすべて使う。
過去の確認trial10–19とは別集合だが、trial0–2は以前の探索で使用済みなので、
盲検の独立確認とは呼ばない。比較方針と論文値は [protocol.md](protocol.md) と
[paper_reference.json](paper_reference.json)、集計は [report.md](report.md) を参照。

`run_experiment.py` は監査adapterをimportせず、本番の `_learn_inter` を
processごとに呼ぶ。各workerへ、標準 `learn` と同じ親seedの子を渡す。
初期proposal cache、通信、更新、全反復CSV、最終checkpoint、native解析は
本番処理を変更しない。親が10試行分の `resolved_config.yaml` を出力rootへ保存する。
初回25反復をtrial0のままprobeとして使い、完了後に共有最大10並列へ広げた。
BLAS/OpenMP/MKLは各process1threadに固定した。

## 保存物

- `execution_manifest.json`: HEAD・本番source/config/実行scriptのSHA256、408入力fileの
  SHA256、前処理後の観測/正解/注視のfingerprint、runtime、CPU/cgroup/メモリ、
  全trialのcommand・seed spawn key・log・時刻・終了code。
- `trial_status/exp000.json`–`exp009.json`: trial単位の時刻・終了状態と
  最終checkpoint/metadataのSHA256、θ/π正規化確認。
- `config.yaml` と `run_experiment.py`: 実際に使用した設定と実行コード。
- `postrun_validation.json`: 実行後のsource/config/scriptと全入力fingerprint再確認。
- `results.json` / `trial_metrics.csv`: 保存状態からの独立集計。

大量のネイティブ出力はgitignore対象の
`outputs/inter_original_heuristics_12ef90f/exp000`–`exp009` に置く。
各trialには両主体の `state_*.npz` / `metadata_*.json`、500反復分のCSV、
`analysis/metrics.json` と `analysis/history.csv`、図を保持する。
`outputs/inter_original_heuristics_12ef90f/logs/` にstdout/stderrがある。

実行環境は Python3.11.18、NumPy2.4.6、SciPy1.17.1、scikit-learn1.9.0。
実行scriptは Ruff0.15.21でcheck PASS。
Ruff0.17.5では同じscriptに14件のstyle指摘（I001、UP017、C408、PLW1510等）が
ある。実行済みscriptとmanifestのSHAを保持するため事後整形はしない。
これは本番sourceに対する別の既存指摘12件と区別する。

## 同じ本番コミットで新規に再実行する

完成済みのartifactやnative出力の上では再実行しない。上書き・自動再開機能はない。
別の**新しい12ef90f worktree**を作り、ここから実行scriptとconfigだけをコピーする。
以下は、このartifactを持つリポジトリrootから実行する例。
実データは同じ内容のものを用意し、パスを環境に合わせて指定する。

```bash
source_repo="$PWD"
new_worktree=/absolute/path/to/new-native-experiment
artifact=docs-for-developper/experiments/inter_original_heuristics_12ef90f
python311=/tmp/csl-audit-python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11
site_packages=/home/horie_takafumi/workspace/multi_agent_csl/legacy-csl/.venv/lib/python3.11/site-packages
new_data_root=/absolute/path/to/datadump

git worktree add --detach "$new_worktree" 12ef90f
mkdir -p "$new_worktree/$artifact"
cp "$source_repo/$artifact/run_experiment.py" "$new_worktree/$artifact/"
cp "$source_repo/$artifact/config.yaml" "$new_worktree/$artifact/"
cd "$new_worktree"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$new_worktree/outputs/mpl-cache"
export PYTHONPATH="$site_packages:$new_worktree/src"
"$python311" - "$artifact/config.yaml" "$new_data_root" "$new_worktree/outputs" <<'PY'
from dataclasses import replace
from pathlib import Path
import sys
from legacy_csl.config import load_config, dump_resolved_config
path = Path(sys.argv[1])
config = load_config(path)
config = replace(config, paths=replace(config.paths,
    data_root=Path(sys.argv[2]), output_root=Path(sys.argv[3])))
dump_resolved_config(config, path)
PY

"$python311" "$artifact/run_experiment.py" --max-parallel 10 \
  > "$new_worktree/supervisor.log" 2>&1
```

CPU/メモリ/IOに応じて `--max-parallel 1`–`10` を指定できる。
並列数はtrialごとのseedや本番遷移を変更しない。
scriptはHEADを12ef90fに固定検証するので、後日のdocs-only commitを含むHEADでは
そのまま動かない。新worktreeには既存のmanifest、trial_status、native出力を
コピーしない。`config.yaml` の変更先も新規出力rootにする。
workerは `start_new_session=True` とファイルlogで起動するが、supervisor自体は
完了まで動かしておく。ホスト停止や実行基盤による強制終了への永続性は保証しない。

## 保存状態の再集計

`aggregate_results.py` は本番学習を再実行せず、保存されたnative outputsを読む。
原データも必要。artifact側のconfig/manifestとnative出力rootを指定する。

```bash
"$python311" "$source_repo/$artifact/aggregate_results.py" \
  --config "$source_repo/$artifact/config.yaml" \
  --output-root /absolute/path/to/outputs/inter_original_heuristics_12ef90f \
  --manifest "$source_repo/$artifact/execution_manifest.json"
```

各CLIの `--help` で引数を確認する。
元の500反復全CSVからの独立照合を含むため、集計にも時間がかかる。
