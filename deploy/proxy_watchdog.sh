#!/bin/bash
# 代理看门狗: 经 mihomo 访问币安 ping, 连续3次失败则重启 mihomo。
# 目的: 覆盖"进程活着但代理不转发"的挂死场景(进程崩溃已由 systemd Restart 覆盖)。
STATE=/home/cc/quant_8.25/ops/.wd_fails
mkdir -p "$(dirname "$STATE")"
code=$(curl -s -m 8 -x http://127.0.0.1:7890 -o /dev/null -w "%{http_code}" https://www.okx.com/api/v5/public/time)
if [ "$code" = "200" ]; then
    echo 0 > "$STATE"
    exit 0
fi
n=$(cat "$STATE" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$STATE"
echo "$(date '+%F %T') 代理健康检查失败($code) 第${n}次" >> /home/cc/quant_8.25/ops/watchdog.log
if [ "$n" -ge 3 ]; then
    echo 0 > "$STATE"
    echo "$(date '+%F %T') 连续3次失败, 重启 mihomo" >> /home/cc/quant_8.25/ops/watchdog.log
    sudo -n systemctl restart mihomo
fi
