"""Authored prose around phase/action chart labels."""

ZH_CN = {
    "{action} ({phase})": "{action}（{phase}）",
    "{action}  ({duration:.1f}s)": "{action}（{duration:.1f} 秒）",
    "<b>Step {index}</b><br>Phase: {phase}<br>Action: {action}<br>Duration: {duration:.1f}s<br>Tokens: {tokens:,}":
        "<b>步骤 {index}</b><br>阶段：{phase}<br>操作：{action}<br>耗时：{duration:.1f} 秒<br>Token 数：{tokens:,}",
    "<b>Step {index} — user prompt</b><br>{preview}": "<b>步骤 {index} — 用户提示</b><br>{preview}",
    "<i>user prompt</i>": "<i>用户提示</i>",
    "user prompt": "用户提示",
}
