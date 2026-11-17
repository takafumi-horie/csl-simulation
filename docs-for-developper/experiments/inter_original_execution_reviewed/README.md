# レビュー後の原典互換INTER・10試行

[結果報告](report.md)が読み始める場所である。実行コードは `2af44df`。
10試行×500反復が完了し、固定原典参照との最終状態400項目・ARI160値が一致した。
結果JSONとCSV、設定、出自manifest、各試行の終了状態をこのディレクトリに保存する。
本番出力は `outputs/inter_original_execution_reviewed/` にあり、反復CSVを含め約4.9GB。

## 再実行

以下は環境に依存しない手順である。[protocol.md](protocol.md)は実行時に固定した
記録として保存しており、そのコードブロックには当時の実行環境のパスも残る。
新しく実行する場合は以下のプレースホルダを実際の環境へ置き換える。
通常の依存関係を導入済みのPython 3.11を使う。

```bash
repo=/path/to/legacy-csl
python311=/path/to/python3.11
site_packages=/path/to/environment/lib/python3.11/site-packages
data_root=/path/to/datadump
artifact=/path/to/new-experiment-records
output_parent=/path/to/new-native-outputs
scripts="$repo/docs-for-developper/experiments/inter_original_execution_reviewed"
export PYTHONPATH="$site_packages:$repo/src"
export MPLCONFIGDIR="$output_parent/matplotlib-cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python311" "$scripts/run_experiment.py" \
  --data-root "$data_root" --artifact-dir "$artifact" \
  --output-parent "$output_parent" --max-parallel 10
```

出力先には新しいディレクトリを指定する。準備だけを行う場合は
`--prepare-only` を追加する。完成済みのmanifest・集計を上書きしない。
最大並列数は1〜10で指定でき、全試行はそれぞれ独立したseedで実行される。
スレッド数は各workerで1に固定し、実際のthreadpool情報も本番出自記録へ保存する。

## 保存済み結果の独立集計

原典参照のアーカイブとsummaryは既存の調査成果物を使う。
パスは引数で指定し、参照ファイル自体は変更しない。

```bash
reference_archive=/path/to/mhr_10_artifacts.tar.gz
reference_summary=/path/to/mhr_10_summary.json
summary_destination=/path/to/new-summary
"$python311" "$scripts/aggregate_results.py" \
  --manifest "$artifact/execution_manifest.json" \
  --reference-archive "$reference_archive" \
  --reference-summary "$reference_summary" \
  --destination "$summary_destination"
```

既存の今回の記録を再集計する場合は、`artifact` をこのディレクトリへ設定する。
manifestに記録した数値環境・入力・コードのhashが一致する必要があり、後から
実装を変更したcheckoutであれば実行コミット `2af44df` のコードを参照する。
集計は学習を再実行せず、native checkpointと正解ラベルからARIを独立に計算する。
データ、native出力、原典参照アーカイブは実行時の出自で場所を確認できるが、
この手順自体に特定マシンのホームや一時ディレクトリへの依存はない。

## 保証と記録

数値配列と、保存形式の違いを対応付けた語・属性・語彙の状態内容を照合した。
全500反復の中間状態を原典と照合した検証ではない。論文の評価対象とA/B集約は
未確定であり、原典参照との一致を論文ARI値の完全再現とは呼ばない。
SINGLE専用原典も未確認で、このINTERの比較はSINGLE原典との照合を代替しない。

[execution_manifest.json](execution_manifest.json)は実行コード・入力・設定・環境、
[results.json](results.json)は独立集計と比較元の出自、
[postrun_validation.json](postrun_validation.json)は最終状態の照合、
反復ファイルの存在確認、RNG記録の実測容量を記録する。
初回実装へのレビューと追加対応は
[レビュー対応記録](../../verification/original_compatibility/review_followup.md)を参照。
