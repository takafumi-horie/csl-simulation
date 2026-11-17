# `current` の後方互換性検証

`defaults_run.py` は変更前コミット `455b499` のソースを `git archive` で
新しい出力ディレクトリへ展開し、以下をそれぞれ独立したPythonプロセスで実行する。
作業ツリー・index・既存の監査checkoutは変更しない。

1. 旧版（`execution_profile` なし）
2. 新版（`execution_profile` 省略）
3. 新版（`execution_profile: current` 明示）

通常条件は両モードについて、importance + `legacy_theta_sum`、Dirichlet +
`legacy_theta_sum` / `none` / `empirical_unigram` の計8条件。
さらにSINGLE/INTER各1条件で従来の4機能をすべて有効にする。
後者はθ100標本平均・raw学習θ・占有Gaussianのμ100標本平均・空カテゴリの
一様μを使う以前のB2構成であり、新しい `original` 実行プロファイルではない。

全10条件はseed21、各属性2カテゴリ、語彙8の同じcompact fixtureを使う。
初期化直後と4更新後までの計5境界を保存する。INTERは通常経路の独立Generator・
初期proposal cache・半ターン順を維持する。SINGLEは固定観測語を使う。
旧版と各新版について、通常条件60対＋4機能条件15対の計75 agent-state対、
1650配列対のdtype・shape・raw bytes、75 metadata対、75 canonical RNG JSON対を
完全一致で比較する。両候補の総計は150 state対・3300配列対となる。
通常の `examples/single.yaml` と `examples/inter.yaml` も旧版とのbyte一致を要求する。

ここでmetadataとは状態保存用の `metadata_*.json` に相当する属性列・語彙を指す。
`learn()` やCLIを通した全出力の比較ではなく、モデル層の軌道と状態内容の
比較である。`validate_data` の処理順、`analyze()` / `predict()` のloader選択は
この旧版対照に含まれない。これらは通常の結合テストとコードレビューで確認した。
`resolved_config.yaml` には新しく `execution_profile: current` が保存されるため、
旧版とファイル全体が同一であるという意味ではない。

これは小規模な短い軌道の後方互換性検証であり、原典との独立照合・実データでの
長期比較・論文ARI値の再現を認定するものではない。本番パッケージや通常テストは
この検証スクリプトへ依存しない。実データ・原典ソース・外部監査adapterも不要。

## 再実行

リポジトリrootで実行する。出力先は存在しない新しいパスを指定する。
Python実体と依存ライブラリの場所は引数で渡し、スクリプト内に一時パスへの依存はない。
`--dependency-path` は必要に応じて複数回指定でき、未指定なら指定Python自身の
site-packagesを使う。`PYTHONPATH` の継承を避け、比較対象のsrcを明示的に選択する。
3つの子プロセスともBLAS/OpenMPを1 threadへ固定する。

```bash
python311=/path/to/python3.11
site_packages=/path/to/environment/lib/python3.11/site-packages
scripts=docs-for-developper/verification/original_compatibility
verify_parent=$(mktemp -d)
"$python311" "$scripts/defaults_run.py" \
  --python "$python311" --dependency-path "$site_packages" \
  --output-root "$verify_parent/defaults"
```

`result.json` に比較結果・旧版commit・依存版・通常YAMLのhash・新版全モジュールと
検証スクリプトのSHA256を保存する。各境界の配列・metadata・RNGと3プロセスのlogは
同じ出力ディレクトリへ残す。`defaults_compare.py` は保存済みの3軌道だけを再比較する
用途にも使える。各スクリプトの `--help` で引数を確認できる。

保存した [defaults_result.json](defaults_result.json) はこの手順の実行結果である。
保存結果の環境はPython 3.11.18、NumPy 2.4.6、SciPy 1.17.1、scikit-learn 1.9.0。
3スクリプトの `--help` と一式の再実行、Ruff 0.15.21の `check`、
`git diff --check` を確認した。
ソースSHAはその実行時点のもので、後のコメント・docstring変更でも変わる。
再実行では状態・metadata・RNGの一致と比較件数を確認し、結果JSON全体やSHAの
同一性を要求しない。依存版が違う場合のbyte一致は保証しない。
