# 原典実行プロファイルの検証記録

2026年11月17日JST。[実装仕様](../../original_compatibility_spec.md)に対する
実装者側の検証資料である。`54a09d7` はユーザー指名の独立レビュワから
ブロッカーなしの承認を受けた。以下は初回実装の記録であり、レビュー後の追加検証は
[レビュー指摘への対応記録](review_followup.md)を参照。

## 保存した結果

| 対象 | 結果 | 記録 |
|---|---|---|
| 原典readerと本番reader | A/Bの全4属性でdtype・shape・bytesが一致。物体数・注意対象も一致。正解ラベルは通常版と同じ | [comparison.json](results/comparison.json) |
| 原典翻訳と本番カーネル | seed 0・2・10、初期化と4反復の75境界。3600状態配列、24提案配列、75共有RNG状態が完全一致 | 同上の `trials[].boundaries` |
| 本番runnerの保存結果 | 同じ3試行の最終状態144配列、初期化・各半ターンの27 RNG境界が照合済み。各試行408入力ハッシュも再検証 | 同上の `trials[].native_runner` |
| 通常版・従来4機能版の互換性 | モデル層の旧 `455b499` 対、新キー省略／`current` 明示が各75状態・1650配列・状態metadata・RNGで完全一致 | [defaults_result.json](defaults_result.json)、[手順と比較範囲](defaults_README.md) |
| 全テスト | 204通過。SINGLEの固定語・触覚・ハイパラ、両モードの保存／解析、INTER予測、設定の拒否条件等を含む | [checks.json](checks.json) |
| 静的検査 | Ruff 0.15.21 `check .` と `git diff --check` が通過 | 同上 |

環境はPython 3.11.18、NumPy 2.4.6、SciPy 1.17.1、scikit-learn 1.9.0。
BLAS/OpenMPは1スレッドに固定。実行時の本番全モジュール、設定、比較スクリプトの
SHA-256を結果JSONに記録している。別ライブラリ版・CPU等でのbyte一致を
保証するものではない。テスト警告10件は、小さい合成データで真値・予測が
単一クラスとなった際のscikit-learnの混同行列／κ係数の警告である。

SINGLE専用Python 2原典は手元になく、SINGLEについて原典全体との照合済みとは
主張しない。単一標本の厳密なGibbs更新、当時のPython 2の軌道、論文ARI値の
完全再現を保証する検証でもない。

## 独立な原典参照

[reference_oracle.py](reference_oracle.py) は本番の学習・初期化関数を一切借りず、
指定された原典をhash確認して翻訳・実行する検証専用コードである。
[compare_original.py](compare_original.py) がその結果を本番カーネルと照合し、
`--native-output` 指定時は通常の本番 `learn()` でも学習・保存を実行して比較する。
本番パッケージと通常のテストは、これらの検証スクリプトや原典に依存しない。

参照は過去の監査で使ったものと同じ現代化規則を固定している。

- Python 2構文の翻訳、`xrange`、触覚ブロックの整数除算、logsumexpの移行。
- 無指定seedを明示seedに置換。KMeansの `n_init=10` を明示する。
- 整数語彙を昇順に固定し、語IDとθの列の対応を明確にする。
- 8箇所のμ100標本生成を `size=100` にまとめる。実際に100標本を引いて平均する。
- 初期化、提案語生成後、MH受理後、各主体のθ更新後に観測hookを挿入する。
  hookはコピーと記録だけを行い、乱数を消費しない。
- readerによる診断ファイル書込みはメモリへ捨て、入力ファイルを変更しない。
  学習の保存関数も検証側で無効化する。原典側はglobal NumPy RNGを使うため、
  検証終了時に呼出し前のglobal RNGを復元する。本番側は最初からローカルRNGを使う。

原典3ソースは[参照ソースのアーカイブ](reference_sources.tar.gz)に保存し、
hashは `reference_oracle.py` と結果JSONに記録している。
翻訳済みの各seedのソースも同じアーカイブの `translated/translated_seed*.py.txt` に
保存した。参照ソースの末尾空行も含めて実行時のbytesを保持している。
原典プログラム名の略称を学習方式の名称には使わない。

## 再実行

通常の依存関係を導入したPython 3.11環境で、リポジトリrootから実行する。
原典ソースは上のアーカイブを展開して使うこともできる。実データは別途必要。
`--native-output` は存在しない新しいディレクトリを指定する。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3.11 docs-for-developper/verification/original_compatibility/compare_original.py \
  --original-root /path/to/original_sources \
  --data-root /path/to/datadump \
  --config examples/inter_original_heuristics.yaml \
  --output /path/to/new-verification-results \
  --native-output /path/to/new-native-outputs \
  --seeds 0 2 10 --iterations 4
```

全配列はdtype・shape・raw bytesを比較する。最初の不一致で例外となり、
差分配列を `first_mismatch.npz` へ保存する。成功時のJSONには各境界の
全配列のhashを残し、単に最終ARIが近いことを合格条件にはしない。
各seed25境界は、初期化1個と、4反復それぞれの2方向×3境界である。
共有RNGは内部keys・位置・正規乱数cacheの有無と値を照合する。

本番runnerによる短い実行の全出力は、再実行コマンドの `--native-output` で
指定した場所に保存される。検証の根拠と再実行手順はこのブランチ内の
スクリプトとJSONに保存してあり、作業マシンの一時ディレクトリを必要としない。

## レビューへの引き渡し

基準 `455b499` との差分を確認する。通常の `single.yaml` / `inter.yaml` は
バイト単位で未変更で、追加の原典設定例2ファイルだけに新profileを指定した。
以前の4機能だけを有効にしたい場合は `execution_profile: current` に戻す。
INTERの原典プロファイルは前処理前のlegacy入力を必要とし、compact NPZのみでは
エラーにする。SINGLEは既存のU前処理と観測語を維持する。

新しいカーネルは `src/legacy_csl/original.py`、接続はconfig/data/model/runnerの
変更箇所を参照。特に共同初期化順、GEM初期πと単一標本の学習π、初期Σ0.05と
GIW事前スケール0.01、fresh proposalと共有RNGの寿命を確認していただきたい。
