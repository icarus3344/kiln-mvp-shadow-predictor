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

从干净环境复现阶段 A 的单一命令：

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/python -m pip install -r requirements.txt && cp config.example.json config.json && .venv/bin/python src/kiln_mvp.py --config config.json
```

运行前先将 `config.example.json` 复制为本机配置 `config.json`，并将 `data_path` 指向本地只读 Parquet；`config.json` 不纳入版本库。原始文件必须保持只读。

GitHub 同步边界：仓库只包含可复现的代码、测试、说明和配置模板；原始 Parquet、虚拟环境、缓存、模型、报告和 `runs/` 实验目录均不上传。

## 输出

运行后在 `artifacts/` 生成：

- `metrics.json`：时间测试集指标；
- `feature_contract.json`：有序特征列、缺失处理和禁用字段；
- `latest_shadow_signal.json`：最新影子预测和目标偏差方向；
- `report.html`：便于查看的 MVP 报告；
- `material_cluster_profiles.csv`：原料聚类中心画像；
- `predictions_tail.csv`：测试集末尾预测；
- `models/*.joblib`：模型及预处理器；
- `minute_cache.parquet`：一分钟数据缓存，后续运行会复用；
- 运行目录中的 `run_manifest.json`：源码/配置/输入哈希、环境、特征清单和生成文件清单。

## 安全边界

- `latest_shadow_signal.json` 只给出“二次风温需要升高/保持/降低”的目标方向，不把相关性模型包装成设备控制策略。
- 在完成受约束模型辨识、反事实验证和现场审批前，不输出煤量、窑速、风机的实际调节值。
- 模型指标只采用严格时间切分和按周滚动测试；训练边界按目标最长窗口留出 purge 间隔，禁止随机行切分。
- 原料聚类的缺失处理、缩放器和聚类中心只在训练期拟合，成分只允许因果 `ffill`，不使用 `bfill`。
- 游离钙以真实值变化事件为样本，不把向前填充的分钟或 5 秒记录当作独立标签；事件特征用延迟后的 `pad` 对齐，不向未来取最近值。
- `窑况日志`、窑况/趋势标签、RTO 推荐和目标派生字段不进入基础特征；窑况标签保留日志/规则衍生疑点。
- 窑况同时输出 30 分钟中心时刻和 25–35 分钟窗口最差两种标签评估，信号字段分别标明，不混用。
- 模型未通过滚动验收时，目标方向信号为 `null`，执行器建议始终为 `null`。

## Luna 后续实施顺序

1. 先复跑当前基线并固定数据质量报告；
2. 加入 LightGBM/CatBoost，与 Ridge/逻辑回归做同一时间测试；
3. 增加滚动回测、按原料聚类分组指标和概率校准；
4. 建立操纵量到预测目标的动态响应模型；
5. 只在历史安全范围内做候选方向搜索；
6. 增加本地 API/仪表板和定时推理；
7. 影子运行通过后，再讨论人工确认式推荐。
