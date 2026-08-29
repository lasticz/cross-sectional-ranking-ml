# quant_8.25 — 个人加密货币量化项目

基于 [Freqtrade](https://github.com/freqtrade/freqtrade)（2026.7）框架。

## 架构

- **本机（Windows）**：策略开发、历史数据下载、回测、hyperopt 调参。
- **服务器（Linux）**：Docker 跑 `docker-compose.yml`，dry-run（模拟盘）→ 小额实盘。
- 策略代码放 `user_data/strategies/`，通过 git 在本机与服务器间同步；数据与数据库不进 git。

## 环境

- conda env：`quant`（Python 3.12），位于 `C:\Users\18970\.conda\envs\quant`
- freqtrade 可执行文件：`C:/Users/18970/.conda/envs/quant/Scripts/freqtrade.exe`
- 建议：`conda activate quant` 后直接用 `freqtrade`

## 常用命令

```bash
# 下载数据（BTC/ETH，config.json 中的 pair_whitelist）
freqtrade download-data --userdir user_data --config config.json --timerange 20230101- --timeframe 1h 4h 1d

# 回测（已含手续费；--slippage 可加滑点假设）
freqtrade backtesting --userdir user_data --config config.json --strategy MyFirstStrategy --timeframe 1h --timerange 20240101-

# 参数优化
freqtrade hyperopt --userdir user_data --config config.json --strategy MyFirstStrategy --timeframe 1h \
  --timerange 20240101-20250601 --hyperopt-loss SharpeHyperOptLoss -e 200

# 本地模拟盘（dry-run，无需 API key）
freqtrade trade --userdir user_data --config config.json --strategy MyFirstStrategy
```

## 服务器部署

1. 服务器装 Docker，clone 本仓库，把真实 `config.json`（含 API key / Telegram token）放到 `user_data/config.json`（不进 git）。
2. `docker compose up -d`，日志在 `user_data/logs/`。
3. 上实盘前：先 `dry_run: true` 跑至少 2-4 周模拟盘，对比回测行为是否一致。

## 流程与纪律

**回测 → hyperopt → 样本外验证 → 模拟盘 → 小额实盘 → 逐步加仓**

- 所有回测结果视为实验性；hyperopt 用前段数据调参，后段数据做样本外验证。
- Freqtrade 回测默认信号在下一根 K 线开盘成交（天然防前视偏差），但自己写 `populate_*` 时不要用未来函数（如 shift(-1)、未来窗口的 rolling）。
- 手续费按交易所实际费率设置（config 中默认用交易所默认值，币安现货 taker 0.1%，有 BNB 抵扣约 0.075%）。

## 目录

```
config.json          # 本机配置（无密钥，dry-run）
config.example.json  # 配置模板
user_data/strategies/  # 策略代码（进 git）
user_data/data/        # 历史数据（不进 git）
docker-compose.yml   # 服务器部署
```
