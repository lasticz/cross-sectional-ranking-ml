#!/bin/bash
# 每日量化系统报告: 收集全部子系统状态与盈亏, 写入 reports/report-YYYYMMDD.md
# 服务器侧自动运行(09:00), 不依赖任何外部机器。
R=~/quant_8.25/reports
mkdir -p "$R"
OUT="$R/report-$(date +%Y%m%d).md"
{
echo "# 量化系统日报 $(date '+%F %T %Z')"
echo
echo "## 1. 服务健康"
echo "- wallet-monitor: $(systemctl is-active wallet-monitor) | mihomo: $(systemctl is-active mihomo)"
docker ps --filter name=freqtrade --format '- {{.Names}}: {{.Status}}'
echo
echo "## 2. 定时器(近24h触发情况)"
systemctl list-timers --no-pager | grep -E 'wallet-refresh|paper-copytrade|funding-scanner|proxy-watchdog'
echo
echo "## 3. 近期错误数(各日志尾2000行内)"
for f in ~/quant_8.25/onchain/monitor.log ~/quant_8.25/onchain/copytrade.log ~/quant_8.25/crossex/scanner.log ~/quant_8.25/user_data/logs/freqtrade-mr.log ~/quant_8.25/user_data/logs/freqtrade-shock.log ~/quant_8.25/user_data/logs/freqtrade-copy.log; do
  n=$(grep -cE '(ERROR|Traceback)' <(tail -2000 "$f") 2>/dev/null)
  echo "- $(basename $f): $n"
done
echo "- 看门狗动作: $(grep -c '重启 mihomo' ~/quant_8.25/ops/watchdog.log 2>/dev/null || echo 0) 次"
echo "- 币安封禁(-1003)记录: $(grep -h 'banned until' ~/quant_8.25/user_data/logs/freqtrade-*.log 2>/dev/null | grep "$(date +%Y-%m-%d)" | wc -l) 次今日"
echo
echo "## 4. 三个 freqtrade bot"
~/quant_8.25/check_bots.sh 2>/dev/null
echo
echo "## 5. 链上监控台账"
echo "- 台账行数: $(wc -l < ~/quant_8.25/onchain/ledger.jsonl 2>/dev/null)"
cd ~/quant_8.25/onchain && /usr/bin/python3 wallet_monitor.py --report 2>/dev/null | head -12
cd ~
echo
echo "## 6. 跟仓模拟"
cd ~/quant_8.25/onchain && /usr/bin/python3 paper_copytrade.py 2>/dev/null | tail -6
cd ~
echo
echo "## 7. 跨所套利 dry-run"
cd ~/quant_8.25/crossex && /usr/bin/python3 funding_scanner.py --status 2>/dev/null
cd ~
echo
echo "## 8. 观察项"
echo "- SPX 跟仓仓是否提前于12h平仓(若提前=custom_exit bug)"
} > "$OUT" 2>/dev/null
echo "报告已写入 $OUT"
