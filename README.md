# ANN 加速的 Leader 分堆

这份代码用于把大量 photo embedding 压缩成半径受控的簇。它包含完整的输入预处理、Faiss HNSW 加速分堆、自动全局半径校准、最终重分配、球面均值中心重算和逐点半径验收。

主程序是单文件 `ann_leader.py`，不依赖同目录下的其他源码。`test_ann_leader.py` 是可单独运行的测试。

## 1. 快速开始

建议使用独立 Python 环境。代码使用 Python 3.10+ 语法；实际验证环境为 Python 3.12.14、NumPy 2.3.5、Faiss CPU 1.16.0。

```bash
python -m pip install -r requirements.txt
python ann_leader.py --help
```

Faiss 的可安装版本取决于 Python、操作系统和 CPU 架构。旧 Linux 系统若遇到 wheel / glibc 不兼容，需要使用兼容的 Faiss 构建或更新运行环境；代码不会自动改用慢速全量搜索。`--backend numpy` 仅供小数据调试。

输入为 `.npy` 文件，形状为 `(N, dimension)`，数据类型为 `float32` 或 `float64`。例如已有一个 NumPy embedding 矩阵时：

```python
import numpy as np
np.save("embeddings.npy", embeddings, allow_pickle=False)
```

程序分块读取和归一化，不会一次复制完整输入到 Python 内存。零向量、NaN、Inf 会报错并给出原始行号，不会静默丢弃 photo。

### 自动选择半径，预算 6 万个簇

```bash
python ann_leader.py \
  --input embeddings.npy \
  --output result_60k \
  --target-clusters 60000 \
  --max-trials 10 \
  --batch-size 128 \
  --threads 8 \
  --center-policy safe_mean \
  --keep-normalized
```

此命令不需要你给每个簇设置半径，也不需要事先猜测一个具体的全局半径。默认在归一化向量的距离尺度上搜索，半径范围为 `[0.0001, 2.0]`。

`60000` 是簇数量的上限预算，不保证恰好产生 60000 个簇。默认在找到簇数位于预算的 95%～100% 的结果时可以提前结束搜索；即使试验次数用尽时未达到该区间，只要有可行方案，也会返回已测试的可行方案，并在报告中说明。

### 预算改为 60 万个簇

第一次运行的 `--keep-normalized` 保留了标准化输入，因此可复用：

```bash
python ann_leader.py \
  --input result_60k/normalized.npy \
  --normalized-input \
  --output result_600k \
  --target-clusters 600000 \
  --max-trials 10 \
  --batch-size 128 \
  --threads 8
```

`--normalized-input` 仍会验证所有输入行，但不再生成一份标准化副本。它要求输入为近似单位长度的 `float32` 向量。

输出目录必须为新目录或空目录，避免覆盖以前的结果。

### 已知半径后，跳过自动校准

```bash
python ann_leader.py \
  --input embeddings.npy \
  --output result_fixed \
  --radius 0.10 \
  --batch-size 128 \
  --threads 8
```

其中 `0.10` 只是命令示例，不能视作对你的数据集的经验结论。

## 2. 距离和分堆规则

每一行先归一化为 `z = x / ||x||`。对理想单位向量，使用：

```text
d(z1, z2) = ||z1-z2||₂ = sqrt(2 - 2*cos(x1, x2))
```

因此这里的 `radius` 是欧氏距离 / 弦距离，不是 `1-cos`，也不是 Faiss 返回的平方距离。若希望用余弦阈值 `tau` 表达，数学上的换算为 `radius = sqrt(2*(1-tau))`。实际代码的验收对象是保存下来的 float32 标准化坐标，并用 float64 差值计算距离。

对每个输入点，程序执行：

1. 从已有固定代表的 HNSW 索引中获取若干候选中心。
2. 用原始标准化坐标重新计算候选距离，避免直接以近似查询结果决定半径验收。
3. 检查当前批次内刚产生、尚未加入全局索引的新中心。
4. 若找到距离不超过 `radius` 的候选，则加入对应簇；否则当前 photo 成为新 leader。

所有 leader 从空集合开始建立，构建过程中不移动。批量查询时，新 leader 对本批后续点立即生效，不会因批查询快照而忽略刚建立的代表。

HNSW 检索可能漏掉合法中心，因此可能多建簇；精确距离验收确保不会将超出半径的点误收入簇。它不保证找到真正最近的中心、不保证簇数最小，也不保证不同 leader 之间都相距大于半径。

## 3. 自动半径如何选择

`automatic_radius_search(...)` 是有试验次数上限的校准过程：

1. 固定数据处理顺序、随机种子、批大小和 ANN 参数。
2. 在给定范围内安排少量对数尺度的粗试验。
3. 每次试验都从空中心开始处理完整数据；一旦准备产生第 `target_clusters + 1` 个中心，就提前停止，因为本轮不会合并已建中心。
4. 找到可行半径后，优先细化当前最小可行半径下方的相邻区间；每四次细化还探索一个更低范围的较宽区间。
5. 在所有已测试且完成全量分配的可行方案中，保存半径最小的一份。
6. 达到目标簇数区间或用尽 `max_trials` 时停止。已保存的完整分配直接进入后处理，不为“最终训练”再重复运行同一试验。

**这是有记录的有限次启发式搜索，不是严格二分，也不证明得到理论最小半径。** 固定顺序的贪心 Leader，其簇数量也可能随半径非单调变化。`radius_search.json` 保存每次试验的半径、是否完成、处理行数、簇数或簇数下界、耗时。

搜索使用全量数据或因超预算而提前停止的数据，不将小样本的簇数按比例外推到 600 万。这样的外推容易受到重复点在抽样中消失的影响。

没有任何已测半径可行时，程序明确报错并保留搜索日志；不会强制把超半径点塞入已有簇。如果你对相似性的容忍有硬上限，请设置 `--radius-max`。例如 `--radius-max 0.15` 表示宁可报告该范围内没有测到可行结果，也不放宽到更大的半径。仅凭期望簇数，程序无法判断业务上“这些图片是否足够相似”。

`max_trials=10` 表示最多 10 次分堆试验，不代表整个程序只扫描数据 10 次。预处理、重分配和中心验收另有扫描。较小半径试验通常会因超预算提前结束，但不能承诺每次都很便宜。

## 4. 最终重分配和中心重算

默认在构建结束后，再用完整的固定 leader 集合做一次 ANN 重分配。每个点的旧中心始终作为保底候选，只有找到严格更近的候选才改变标签。所以该阶段不会增加任何点到所属中心的距离。相同距离保留旧标签。`--no-reassign` 可以省略这一次扫描。

之后，为每个最终簇重新计算球面均值：

```text
s_j = sum(z_i for i in cluster_j)
mean_j = s_j / ||s_j||₂
```

求和使用 float64，并把 `M × dimension` 的累加矩阵放在临时 memmap 文件中。所有均值转为 float32 后，再完整扫描所有簇成员，计算其最大半径。

提供两种策略：

| 参数 | 最终 `centers.npy` | 半径性质 |
|---|---|---|
| `--center-policy safe_mean`（默认） | 均值合法则采用均值，否则保留原 leader | 保持每个点到其最终中心的距离不超过所选半径 |
| `--center-policy mean` | 采用所有定义良好的球面均值 | 可能突破原半径，报告真实最大半径和是否越界 |

**两种策略都会计算所有簇的均值。** `mean_centers.npy` 保存所有计算出的均值；`center_is_mean.npy` 说明最终中心实际采用了哪些均值。

如果某簇向量和恰为零，球面均值不可定义。此时 `mean_defined=False`，`mean_centers` 的该行用旧中心作有限值占位；使用该文件时应同时读取 `mean_defined.npy`。这可以避免 NaN 污染后续距离计算。

均值中心一般不是实际 photo。`leader_ids.npy` 始终保留真实原始 photo 的行号；若下游必须选择真实图片，应通过原始行号和标签保留实际候选，而不要把均值当成一个真实 photo ID。

## 5. 输出文件

| 文件 | 形状 / 内容 | 用途 |
|---|---|---|
| `centers.npy` | `(M, dimension)`, float32 | 最终用于几何计算的中心 |
| `labels.npy` | `(N,)`, int64 | `labels[i]` 为原输入第 i 行的最终簇号 |
| `counts.npy` | `(M,)`, int64 | 每簇实际 photo 数量，总和等于 N |
| `max_radius.npy` | `(M,)`, float64 | 每簇相对于最终中心的实际最大半径 |
| `leader_ids.npy` | `(M,)`, int64 | 每簇原始 leader 在输入矩阵中的行号 |
| `leader_centers.npy` | `(M, dimension)`, float32 | 重算均值前的固定 leader 向量 |
| `mean_centers.npy` | `(M, dimension)`, float32 | 所有球面均值，需配合 `mean_defined` |
| `mean_defined.npy` | `(M,)`, bool | 球面均值是否可定义 |
| `mean_max_radius.npy` | `(M,)`, float64 | 相对于球面均值的实测最大半径；未定义行是占位值 |
| `center_is_mean.npy` | `(M,)`, bool | 最终中心是否采用了均值 |
| `radius_search.json` | 每轮试验记录 | 查看为什么选择这个半径 |
| `report.json` | 参数、规模、半径和耗时统计 | 运行摘要和审计结果 |
| `normalized.npy` | 可选 `(N, dimension)` | 使用 `--keep-normalized` 时保留，供下次复用 |

读取示例：

```python
from pathlib import Path
import json
import numpy as np

out = Path("result_60k")
centers = np.load(out / "centers.npy", mmap_mode="r")
labels = np.load(out / "labels.npy", mmap_mode="r")
counts = np.load(out / "counts.npy", mmap_mode="r")
max_radius = np.load(out / "max_radius.npy", mmap_mode="r")
report = json.loads((out / "report.json").read_text())

print("radius:", report["chosen_radius"])
print("cluster count:", len(centers))
print("radius preserved:", report["radius_preserved"])
print("original row 123 belongs to:", int(labels[123]))
assert int(counts.sum()) == len(labels)
```

不要对每个簇重复扫描全部 `labels` 来收集成员，那会产生 `O(NM)` 额外开销。需要完整的成员倒排表时，可一次按 `labels` 排序，再用 `counts` 的前缀和切片；或者在原有数据库中保存 `photo_id -> cluster_id` 映射。

## 6. 直接作为 Python 模块调用

```python
from tempfile import TemporaryDirectory
from ann_leader import (
    IndexConfig, prepare_embeddings, automatic_radius_search,
    load_leader_result, reassign_to_leaders, recenter_clusters,
)

X = prepare_embeddings("embeddings.npy", "normalized.npy")
config = IndexConfig(
    backend="hnsw", hnsw_m=32,
    ef_search=64, ef_construction=200,
    candidates=8, threads=8,
)

with TemporaryDirectory() as work:
    tuned = automatic_radius_search(
        X, target_clusters=60000, index_config=config,
        work_dir=work, max_trials=10, batch_size=128,
    )
    result = load_leader_result(tuned.best_result_dir)
    result = reassign_to_leaders(X, result, config)
    final = recenter_clusters(X, result, work, policy="safe_mean")
    # 持久化要用到的结果；临时目录退出后会删除搜索缓存。
    # final["centers"]、result.labels、final["counts"] 是核心输出。
```

推荐首先使用 CLI，它会统一保存输出、搜索记录和报告。模块接口假定后续步骤使用与分堆时完全相同的标准化输入，不能在构建后替换或重排 X。

## 7. 600 万数据的运行参数与资源

| 参数 | 默认值 | 含义与取舍 |
|---|---:|---|
| `--hnsw-links` | 32 | HNSW 图连边参数，增加通常需要更多内存和构建工作 |
| `--ef-search` | 64 | 搜索探索量；128 或 256 可用于评估更高召回的效果 |
| `--ef-construction` | 200 | 图构建探索量 |
| `--candidates` | 8 | 每个点从全局索引取回并验真的候选中心数 |
| `--batch-size` | 128 | 分堆批大小；未知半径时保守控制批内新中心的检查量 |
| `--refine-batch-size` | 4096 | 重分配与均值扫描的批大小 |
| `--threads` | 1 | Faiss 线程数；示例用 8，可按 CPU 和实测调整 |
| `--max-trials` | 10 | 自动半径分堆试验上限 |
| `--target-tolerance` | 0.05 | 达到预算的 95%～100% 时允许提前停止 |
| `--order` | `block_shuffle` | 打乱读取块的顺序，块内保持原始行顺序 |
| `--seed` | 42 | 控制输入处理顺序的随机种子 |

提高 ANN 搜索强度可以减少漏召回风险，但不承诺最终簇数严格下降。贪心建中心的后续路径也可能改变。

`--threads 1` 用于更稳定的同环境复现。多线程 HNSW 构建可能存在非确定性；固定种子并不保证跨线程数、库版本、机器都逐位一致。改变 `batch-size` 也会改变 `block_shuffle` 的处理顺序。一轮自动校准内这些参数保持固定。

程序不构建 `N×N` 或 `M×M` 距离矩阵，但内存和磁盘需求并非与数据规模无关。以 dimension=512 为例，按十进制 GB 计算：

- 600 万行 float32 输入或标准化副本：`6,000,000 × 512 × 4 = 12.288 GB`。
- 60 万中心的一份 float32 矩阵：`600,000 × 512 × 4 = 1.2288 GB`；6 万中心为 `0.12288 GB`。
- 60 万中心的 float64 均值累加文件：`600,000 × 512 × 8 = 2.4576 GB`；6 万中心为 `0.24576 GB`。
- 600 万行的 int64 标签：`48 MB`。

HNSW 会额外保存图和自己的向量副本；后处理也会同时保留 leader、所有均值和最终中心等多份矩阵。memmap 可以避免主动一次读入全部输入，但不会消除磁盘读写、操作系统页缓存和实际工作内存。临时文件和输入最好放在本地 SSD；可用 `--work-dir` 指定临时文件根目录。

批内新中心的检查是精确的，极端情况下成本为 `O(N × batch_size × dimension)`。因此批大小不是越大越快，尤其在小半径试验中几乎每点都新建中心时。默认 128 是保守起点；实际数据和半径比较稳定后再测 256 等参数。

这份实现验证了几何和控制流程的正确性，没有在你的 600 万 photo 数据上测量速度，不据此承诺具体分钟数或加速倍数。

## 8. 与后续 MSD / OSG 的衔接

本程序只产生几何压缩和原始行映射，没有替你运行 OSG，也没有改变每张 photo 的质量分、标签或可用性约束。

把某簇近似为中心后，可共享该簇的几何距离计算；仍需保存实际成员与容量 `counts`。一簇只允许选一个候选是额外约束，不能在原问题允许选多个成员时自动加上。`counts` 也不应未经目标函数推导就直接当成质量权重。

对所有成员与最终中心均满足半径 r 的情况，三角不等式给出：

```text
|d(x,y) - d(center(x),center(y))| <= 2*r
```

同簇两张不同 photo 的真实距离可能非零。共享中心距离是可控的近似，不表示原始距离增量完全相等。这也是默认提供 `safe_mean` 和最终半径审计的原因。

## 9. 验证

```bash
python -m unittest -v test_ann_leader
```

交付时 22 项测试全部通过，覆盖：真实 Faiss HNSW / Flat、原始行号映射、同批新增代表、早停、ANN 漏召回、归一化输入验证、重分配保底、均值越界回退、零和均值、半径非单调反例、最小已测可行半径选择以及完整 CLI 输出。

若没有安装 Faiss，相关测试会显示 skipped；使用 HNSW 运行仍会明确要求安装 Faiss，不会静默退化。测试数据是小规模构造数据，并不是生产速度基准。

## 10. 依赖文档

- Faiss 索引和 HNSW 参数：<https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
- Faiss 距离定义、平方 L2 与归一化余弦关系：<https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances>

