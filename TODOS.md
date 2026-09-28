# TODOS

Deferred on 2026-09-25 during the src-layout review. None of these block the move to `ashare-daily`.

## detone 打开时的 MLFAM 路径

- **What:** 选池去噪打开时，不再用 `parents[3]/etf_strategy_MLFAM` 找外部包，改成配置里的一条路径。
- **Why:** 文件进 `src/` 之后，`parents[3]` 不再指向原来的兄弟目录。生产默认去噪是关的，日更不会走到这里。打开去噪时会找不到包。
- **Pros:** 以后打开去噪的人知道从 `clustering.py` 的 `apply_detone_to_corr` 改起。
- **Cons:** 现在用不到。
- **Context:** 生产默认 detone 关闭。这次搬家不改这个回退。不要为本条再写一套 `parents` 向上走。
- **Depends on:** `src/ashare_daily` 布局先落地。不挡住八个子命令。

## fig9 扫描默认读取 d080 聚类 CSV

- **What:** 让图9扫描默认读取 `cluster_representatives_*_d080.csv`。
- **Why:** HRP `--dist-t 0.8` 写出的是 `_d080.csv`。现有扫描默认不读这个后缀，0.8 的聚类进不了图9。
- **Pros:** 差的是默认文件名，不是再跑一次 HRP。
- **Cons:** 扫描器不在本次闭包里，条目指向仓库外的脚本。
- **Context:** 扫描脚本留在旧的 stock 树。这次八个命令不含它。`hrp` 子命令仍把 `_d080` 写到本地 json 的 `TEMP_DIR`。默认距离仍是 0.60，0.60 的文件没有距离后缀。
- **Depends on:** 不挡住这次搬家。做的时候再决定扫描器进不进这个包。

## 日更跑稳后再补 GitHub Actions 和 CITATION.cff

- **What:** 八条 `ashare-daily` 长任务各成功一次之后，再加 GitHub Actions 跑测试，并加 `CITATION.cff`。
- **Why:** 这次只要能 `pip install -e .`、一条命令、测试能导入。Actions 和引用文件是之后的事。
- **Pros:** 「没做 Actions」是故意推迟，不是这次漏了。
- **Cons:** 仓库会有一条还没做的 CI 说明。
- **Context:** 安装方式是 clone 之后 `pip install -e .`。不发 PyPI，不做容器。Qlib 行情数据不在包里。旧树仍是日更入口，直到八条命令各成功一次。
- **Depends on:** 布局验收通过，并且八条长任务在新命令上各成功一次。

## 生产聚类窗口拉到真正的 252 日

- **What:** 等召回率审计出数之后，把 `configs/shared_filter_config.json` 的 `test_period` 起点前移，让日更 `cluster-map` 的面板长于 253 行，从而用上 `CLUSTER_LOOKBACK_DAYS=252`，而不是现在约 244 个交易日的整段。
- **Why:** `prepare_recent_window` 要 253 行。当前 `test_period` 是 2025-06-30 到 2026-06-30，短于这个尾巴，生产实际把整段短面板拿去聚类。审计器评的是真正的 252 日。两套池子要用 Jaccard 对照，不能在审计完成前改日更。
- **Pros:** 日更池子和被审计的算法变成同一个窗口，召回率结论才能直接指导 `POOL_TARGET_COUNT`。
- **Cons:** 改配置会改变每天的池子。审计没出数之前改，测量对象和生产同时变。
- **Context:** 2026-09-28 eng review D1/D16/D21。审计器故意不改这条配置。summary JSON 里的 Jaccard 是差距的证据。从 `shared_filter_config.json` 的 `test_period[0]` 改起，不要改 `prepare_recent_window`。
- **Depends on:** `recall_audit` 全量跑完，并且 summary 里已经有和生产池的 Jaccard。

## 召回率的 h 与 N 敏感性（不改判定带）

- **What:** 在第一份 `status: ok` 的 summary 之后，另跑 h∈{5,10,20}、赢家人数∈{20,50}，只比较结论的排序是否和 h=10、N=50 一样。不把扫描结果写进判定带。
- **Why:** 判定带已经锁在 h=10、前 50 名。若换一个持有期或赢家人数，档位会翻，这份 JSON 就不能单独指导要不要加目标只数。
- **Pros:** 用同一套审计器回答「结论稳不稳」，不必新写选股器。
- **Cons:** 每多一档就要再聚类一轮，墙钟时间大约成倍增加。
- **Context:** 2026-09-28 autoplan CEO E2。判定均值仍只用 h=10 的不重叠锚点。扫描是附录，不是第二条判定规则。
- **Depends on:** 全量审计 JSON 已经是 `status: ok`。

## 不要在审计 JSON 之前改生产目标只数

- **What:** 在 summary 给出档位之前，不要把 `POOL_TARGET_COUNT` 改成 150 或 200 去做「顺便看一眼 HTML 会不会变大」。
- **Why:** 目标只数一改，日更池子和被测量的算法就不是同一个。审计要回答的问题会作废。
- **Pros:** 明天的 100 只观察池保持不动，档位仍然可比。
- **Cons:** 不知道下游 HTML 在 150 只时有多大，要等审计结束再量。
- **Context:** 2026-09-28 autoplan CEO E6。操作者本意是先测量再改。这一条是把那条约束写进待办，避免以后的改动把它当成可以提前做的探针。
- **Depends on:** 全量审计 JSON 的档位，加上和生产 CSV 的 Jaccard。
