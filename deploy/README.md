# 服务器 dry-run 部署（192.168.1.100）

## 当前运行

| 容器 | 策略 | 端口(仅本机) | 数据库 | 日志 |
|---|---|---|---|---|
| freqtrade-mr | MeanReversionStrategy (15m) | 127.0.0.1:8083 | tradesv3-mr.sqlite | user_data/logs/freqtrade-mr.log |
| freqtrade-shock | BtcShockStrategy (15m) | 127.0.0.1:8084 | tradesv3-shock.sqlite | user_data/logs/freqtrade-shock.log |

两 bot 均为 dry-run（150U 模拟、27 币、10% 保证金 × 3x、maker 费模型），`restart: unless-stopped`。

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
