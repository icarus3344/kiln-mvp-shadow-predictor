# 窑系统预测 MVP

这是一个只读、离线、影子模式的最小可运行版本。它不会连接 DCS，也不会下发控制指令。

当前 MVP 会完成：

1. 将 5 秒 Parquet 数据汇总到 1 分钟；
2. 识别稳定生产工况并清理明显坏值；
3. 对出磨生料成分做 K-Means 聚类；
4. 用 Ridge 预测未来第 25–35 分钟二次风温均值；
5. 预测二次风温上升/稳定/下降趋势；
6. 预测 30 分钟后窑况好/中/差；
7. 按真实化验变化事件预测游离钙及其趋势；
8. 生成 JSON、CSV、模型文件和静态 HTML 报告。

## 运行

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.example.json config.json
# 编辑 config.json，将 data_path 指向本机只读 Parquet
.venv/bin/python src/kiln_mvp.py --config config.json
```

从干净环境运行的单一命令：

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/python -m pip install -r requirements.txt && cp config.example.json config.json && .venv/bin/python src/kiln_mvp.py --config config.json
```

运行前先将 `config.example.json` 复制为本机配置 `config.json`，并将 `data_path` 指向本地只读 Parquet；`config.json` 不纳入版本库。原始文件必须保持只读。

## 输出

运行后在 `artifacts/` 生成：

- `metrics.json`：时间测试集指标；
- `feature_contract.json`：有序特征列、缺失处理和禁用字段；
- `latest_shadow_signal.json`：最新影子预测和目标偏差方向；
- `report.html`：便于查看的 MVP 报告；
- `material_cluster_profiles.csv`：原料聚类中心画像；
- `predictions_tail.csv`：测试集末尾预测；
- `models/*.joblib`：模型及预处理器；
- `minute_cache.parquet`：一分钟聚合缓存；
- `fcao_event_alignment.csv`：游离钙真实变化事件与预测时刻对齐明细；
- `fcao_delay_alignment.json`：游离钙分组延迟候选、训练期选择和可用性规则；
- `inlet_chemistry_alignment_audit.csv`：出磨到入窑 KH/SM/IM 的探索性时移相关性审计；
- `stage_a2_pre_post_comparison.json`：时间对齐整改前后按周逐折指标对照；
- 运行目录中的 `run_manifest.json`：源码/配置/输入哈希、环境、特征清单和生成文件清单。

## 安全边界

- `latest_shadow_signal.json` 只给出“二次风温需要升高/保持/降低”的目标方向，不把相关性模型包装成设备控制策略。
- 在完成受约束模型辨识、反事实验证和现场审批前，不输出煤量、窑速、风机的实际调节值。
- 模型指标只采用严格时间切分和按周滚动测试；训练边界按目标最长窗口留出 purge 间隔，禁止随机行切分。
- 一分钟桶定义为 `[bucket_start, bucket_end)`；过程均值只在 `bucket_end` 后使用，`prediction_time` 等于 `bucket_end`，未来窗口从 `prediction_time` 计算。
- 原料聚类的缺失处理、缩放器和聚类中心只在训练期拟合，成分只允许因果 `ffill`，不使用 `bfill`。
- 游离钙以真实值变化事件为样本，不把向前填充的分钟或 5 秒记录当作独立标签；上一化验值必须严格早于 `prediction_time`，事件特征按变量组独立选择延迟并用因果 `pad` 对齐，不向未来取最近值。
- 游离钙的生产时间、取样时间和实验室结果可用时间在当前数据中均标记为未知；historian 观测时间不被解释为生产或取样时间。
- `窑况日志`、窑况/趋势标签、RTO 推荐和目标派生字段不进入基础特征；窑况标签保留日志/规则衍生疑点。
- 窑况同时输出 30 分钟中心时刻和 25–35 分钟配置化窗口规则标签，当前窗口规则未声称获得现场认可，信号字段分别标明，不混用。
- 模型未通过滚动验收时，目标方向信号为 `null`，执行器建议始终为 `null`。
