"""Simplified Chinese for model settings and credential-safe connection status."""

ZH_CN = {
    "Testing connection…": "正在测试连接…",
    "Not tested. Sends a small model request; no trajectory data. Testing does not apply settings.":
        "尚未测试。测试会发送一个小型模型请求，不包含轨迹数据，也不会应用设置。",
    "The connection test could not be completed. Please retry.": "无法完成连接测试，请重试。",
    "Settings applied for this session.": "已为当前会话应用设置。",
    "Connection incomplete; enter the missing connection fields before requesting analysis.":
        "连接信息不完整，请补充缺失字段后再请求分析。",
    "API keys are session-only. No model request was sent.": "API 密钥仅保存在当前会话中。未发送模型请求。",
    "Using environment defaults. Remembered connection details cleared.": "正在使用环境默认值。已清除记住的连接信息。",
    "Using environment defaults. API keys stay in the current session.": "正在使用环境默认值。API 密钥仅保存在当前会话中。",
    "Saved connection details were invalid; using environment defaults.": "保存的连接信息无效，正在使用环境默认值。",
    "Connection details restored. Enter your API key and apply settings to enable analysis.":
        "已恢复连接信息。请输入 API 密钥并应用设置，以启用分析。",
    "Leave blank to keep the active key for this endpoint. Session only.":
        "留空可保留此接口当前使用的密钥。密钥仅保存在当前会话中。",
    "Includes hidden thinking. Increase if replies are empty or cut off.":
        "包含模型内部思考消耗。若回复为空或被截断，请提高此值。",
    "API keys stay in this session and are never remembered.": "API 密钥仅保存在当前会话中，不会被记住。",
    "Ready": "已就绪",
    "sends to": "发送至",
    "UI settings": "界面设置",
    "using LABEL_*": "使用 LABEL_*",
    "Not configured": "未配置",
    "set base URL in Settings": "请在设置中填写接口地址",
    "set API key in Settings": "请在设置中填写 API 密钥",
    "set model in Settings": "请在设置中填写模型",
    "set base URL, API key in Settings": "请在设置中填写接口地址和 API 密钥",
    "set base URL, model in Settings": "请在设置中填写接口地址和模型",
    "set API key, model in Settings": "请在设置中填写 API 密钥和模型",
    "set base URL, API key, model in Settings": "请在设置中填写接口地址、API 密钥和模型",
    "open Settings, or set": "打开设置，或将",
    "or": "或",
    "in the environment or a": "写入环境变量或",
    "file": "文件",
    "open Settings, or set {fields} in": "打开设置，或将 {fields} 写入",
    "{count} steps in context": "上下文中有 {count} 个步骤",
    "load a trajectory first": "请先加载轨迹",
    "Enter Base URL before testing.": "测试前请填写接口地址。",
    "Enter Model before testing.": "测试前请填写模型。",
    "Enter API key before testing.": "测试前请填写 API 密钥。",
    "Enter Base URL, Model before testing.": "测试前请填写接口地址和模型。",
    "Enter Base URL, API key before testing.": "测试前请填写接口地址和 API 密钥。",
    "Enter Model, API key before testing.": "测试前请填写模型和 API 密钥。",
    "Enter Base URL, Model, API key before testing.": "测试前请填写接口地址、模型和 API 密钥。",
    "Enter Base URL, Model and API key before testing.": "测试前请填写接口地址、模型和 API 密钥。",
    "Select OpenAI-compatible or Anthropic before testing.": "测试前请选择 OpenAI 兼容接口或 Anthropic。",
    "Enter a positive Timeout before testing.": "测试前请填写大于零的超时时间。",
    "Connection successful. The API accepted the key and selected model.":
        "连接成功。API 已接受该密钥和所选模型。",
    "Connection successful. The API accepted the key and selected model; the small test token budget was exhausted.":
        "连接成功。API 已接受该密钥和所选模型；本次小型测试的 Token 额度已用完。",
    "The API returned an unexpected response format. Check Provider, Base URL and Model; the key could not be verified.":
        "API 返回了非预期的响应格式。请检查提供商、接口地址和模型；尚无法确认密钥是否有效。",
    "The API responded without a test reply. The key could not be verified; check the selected model and provider.":
        "API 未返回测试回复，尚无法确认密钥是否有效。请检查所选模型和提供商。",
    "The connection test timed out. Retry or check the endpoint; the key could not be verified.":
        "连接测试超时。请重试或检查接口；尚无法确认密钥是否有效。",
    "The API redirected the request. Enter the final Base URL; no redirect was followed.":
        "API 要求重定向。请输入最终接口地址；测试未跟随重定向。",
    "The API rejected the key (HTTP 401). Check the API key.": "API 拒绝了该密钥（HTTP 401）。请检查 API 密钥。",
    "Access denied (HTTP 403). Check the key's permissions and access to the selected model.":
        "访问被拒绝（HTTP 403）。请检查密钥权限以及对所选模型的访问权限。",
    "Endpoint or model not found (HTTP 404). Check Base URL and Model.":
        "未找到接口或模型（HTTP 404）。请检查接口地址和模型。",
    "Rate limit or quota reached (HTTP 429). Retry later or check account limits; the key could not be verified.":
        "已达到速率或配额限制（HTTP 429）。请稍后重试或检查账户限制；尚无法确认密钥是否有效。",
    "Billing or credits are required (HTTP 402). Check the provider account; the key could not be verified.":
        "需要启用计费或充值（HTTP 402）。请检查提供商账户；尚无法确认密钥是否有效。",
    "The API rejected the test request (HTTP {status}). Check Provider, Base URL and model compatibility.":
        "API 拒绝了测试请求（HTTP {status}）。请检查提供商、接口地址和模型兼容性。",
    "The provider returned a service error (HTTP {status}). Retry later; the key could not be verified.":
        "提供商返回服务错误（HTTP {status}）。请稍后重试；尚无法确认密钥是否有效。",
    "The API returned HTTP {status}. The key could not be verified.": "API 返回 HTTP {status}，尚无法确认密钥是否有效。",
    "The TLS certificate could not be verified. Check Base URL and the server certificate.":
        "无法验证 TLS 证书。请检查接口地址和服务器证书。",
    "Could not reach the API. Check Base URL and the network; the key could not be verified.":
        "无法连接到 API。请检查接口地址和网络；尚无法确认密钥是否有效。",
}
