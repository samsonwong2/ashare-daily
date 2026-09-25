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
