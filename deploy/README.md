# 服务器 dry-run 部署（192.168.1.100）

## 当前运行

| 容器 | 策略 | 端口(仅本机) | 数据库 | 日志 |
|---|---|---|---|---|
| freqtrade-mr | MeanReversionStrategy (15m) | 127.0.0.1:8083 | tradesv3-mr.sqlite | user_data/logs/freqtrade-mr.log |
| freqtrade-shock | BtcShockStrategy (15m) | 127.0.0.1:8084 | tradesv3-shock.sqlite | user_data/logs/freqtrade-shock.log |
| nt-ml-dryrun | ML 横截面 Top3/Bottom3 (15m, 8h 调仓) | — | — | docker logs nt-ml-dryrun |

两 freqtrade bot 均为 dry-run（150U 模拟、27 币、10% 保证金 × 3x、maker 费模型），`restart: unless-stopped`。

## NT ML dry-run 节点 (nt-ml-dryrun)

ML v2 横截面策略的 NautilusTrader dry-run：**真实 Binance USDT-M 行情 + sandbox 模拟成交**
（150U 起始、3x 杠杆、reduce-only 止损单生效），不产生真实订单、不需要 API 交易权限。

- 每 8h（00/08/16 UTC 的 15m bar 收盘）用部署模型在线计算横截面排名 → 多 Top3 / 空 Bottom3
- 信号管线与回测决策等价性已由 `scripts/live/validate_features.py` 验证（ALL PASS）
- 状态: `user_data/ml_v2/live/status.json`（净值/持仓/决策计数）、`trades.jsonl`（逐笔平仓）
- 净值续接: 重启后以 `state.json` 的期末净值为起始余额（仓位清零，≤8h 持仓影响小）

```bash
# 首次部署（在服务器 ~/quant_8.25/）
git pull
# 上传模型产物（本地执行）:
#   scp user_data/ml_v2/deploy_{model.pkl,clf.pkl,features.json,meta.json} cc@192.168.1.100:~/quant_8.25/user_data/ml_v2/
docker compose -f deploy/docker-compose-nt.yml up -d --build

# 日常检查
docker logs --tail 50 nt-ml-dryrun
cat user_data/ml_v2/live/status.json
tail -5 user_data/ml_v2/live/trades.jsonl

# 重建/重启
docker compose -f deploy/docker-compose-nt.yml up -d --build   # 代码更新后
docker compose -f deploy/docker-compose-nt.yml restart          # 仅重启（仓位清零）
```

代理依赖同下（mihomo 127.0.0.1:7890，容器内经 host.docker.internal 访问）。

## ⚠️ 代理依赖

服务器无法直连币安，行情流量走**服务器本机的 mihomo**（systemd 服务 `mihomo.service`，127.0.0.1:7890），
容器内通过 `host.docker.internal:7890` 访问（compose 已配 host-gateway）。
- 已加 ufw 规则: 允许 172.17.0.0/16 和 172.23.0.0/16（compose 网段）访问宿主机 7890。
- 依赖: mihomo 服务运行中且节点可用。检查: `systemctl status mihomo` / `curl -x http://127.0.0.1:7890 https://fapi.binance.com/fapi/v1/ping`
- **与 Windows 机器无关**（早期方案曾借道 Windows Clash，已弃用）。

## 常用操作（在服务器 ~/quant_8.25/ 下）

```bash
./check_bots.sh                    # 两个 bot 健康检查（API 状态/持仓）
docker compose ps                  # 容器状态
docker compose logs -f freqtrade-mr
tail -f user_data/logs/freqtrade-mr.log
docker compose restart             # 重启
docker compose down                # 停止
```

## 本机看 FreqUI / API

API 只监听服务器 127.0.0.1，从 Windows 建隧道访问：
`ssh -L 8083:127.0.0.1:8083 -L 8084:127.0.0.1:8084 cc@192.168.1.100`
然后浏览器开 http://127.0.0.1:8083 （用户 freqtrader / mr_dryrun_pw_2026）

## 更新策略

本机改完 `user_data/strategies/*.py` 后：
`scp user_data/strategies/X.py cc@192.168.1.100:~/quant_8.25/user_data/strategies/ && ssh cc@192.168.1.100 "cd ~/quant_8.25 && docker compose restart"`

## 判读标准（dry-run 跑 1-2 个月后）

- 对照 walk-forward 基准: MR 年化 ~17% / 回撤 ~24%；Shock 年化 ~16%（逐段波动大）
- 重点观察 maker 挂单成交率（回测最乐观的假设）与实际滑点
- 两个策略交易分布是否符合各自设计（MR: 布林回归；Shock: 仅 BTC 急跌后 2h 持仓）
