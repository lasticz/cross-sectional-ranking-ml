#!/bin/bash
# 检查所有 freqtrade bot 的 API 状态
# 密码从环境变量读取（在服务器 ~/.bashrc 或 .env 中设置）
# export FT_MR_PW="xxx" FT_SHOCK_PW="xxx" FT_COPY_PW="xxx"

FT_MR_PW="${FT_MR_PW:-$(grep FT_MR_PW ~/.quant_env 2>/dev/null | cut -d= -f2)}"
FT_SHOCK_PW="${FT_SHOCK_PW:-$(grep FT_SHOCK_PW ~/.quant_env 2>/dev/null | cut -d= -f2)}"
FT_COPY_PW="${FT_COPY_PW:-$(grep FT_COPY_PW ~/.quant_env 2>/dev/null | cut -d= -f2)}"

for spec in "MR:8083:${FT_MR_PW}" "SHOCK:8084:${FT_SHOCK_PW}" "COPY:8085:${FT_COPY_PW}"; do
    name=${spec%%:*}
    rest=${spec#*:}
    port=${rest%%:*}
    pw=${rest#*:}
    echo "--- $name (port $port) ---"
    login=$(curl -s -m 8 -X POST -u "freqtrader:${pw}" http://127.0.0.1:$port/api/v1/token/login)
    token=$(echo "$login" | python3 -c 'import sys,json
try: print(json.load(sys.stdin)["access_token"])
except Exception as e: print(""); sys.stderr.write("login resp: " + str(e) + "\n")')
    if [ -z "$token" ]; then
        echo "login failed: $login"
        continue
    fi
    curl -s -m 8 http://127.0.0.1:$port/api/v1/show_config \
        -H "Authorization: Bearer $token" | python3 -c 'import sys,json
d = json.load(sys.stdin)
print("dry_run:", d.get("dry_run"), "| strategy:", d.get("strategy"), "| state:", d.get("state"), "| exchange:", d.get("exchange"))'
    curl -s -m 8 http://127.0.0.1:$port/api/v1/whitelist \
        -H "Authorization: Bearer $token" | python3 -c 'import sys,json
print("whitelist pairs:", len(json.load(sys.stdin).get("whitelist", [])))'
    curl -s -m 8 http://127.0.0.1:$port/api/v1/status \
        -H "Authorization: Bearer $token" | python3 -c 'import sys,json
d = json.load(sys.stdin)
print("open trades:", len(d))
for t in d:
    print("  %s 浮盈 %.2f%% 已持 %smin" % (t["pair"], t["profit_pct"], t.get("trade_duration", 0)))'
done
