import os
import datetime
import subprocess
import tempfile
import json
import sys
import re
import requests
from flask import Flask, request, Response, stream_with_context, render_template
import pymysql
from openai import OpenAI
from bs4 import BeautifulSoup
import redis
import uuid
from flask import request, make_response


app = Flask(__name__)
sys.stdout.reconfigure(line_buffering=True)

MODEL_DISPLAY_NAME = "江洋大模型（联网搜索版）"
ANSWER_PREFIX = f"【{MODEL_DISPLAY_NAME}】"

# ---------- 本地 Ollama 配置 ----------
LOCAL_MODEL_NAME = "qwen2:1.5b-instruct"
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'localhost')
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:11434/api/generate"

redis_client = redis.Redis(
    host=os.environ.get('REDIS_HOST', 'localhost'),
    port=int(os.environ.get('REDIS_PORT', 6379)),
    decode_responses=True   # 自动解码为字符串
)

def call_local_llm(prompt, system_prompt="", stream=False, max_tokens=512):
    """调用本地 Ollama 模型，支持流式和非流式。失败时返回 None。"""
    full_prompt = system_prompt + "\n" + prompt if system_prompt else prompt
    payload = {
        "model": LOCAL_MODEL_NAME,
        "prompt": full_prompt,
        "stream": stream,
        "options": {
            "temperature": 0.0 if not stream else 0.7,
            "num_predict": max_tokens
        }
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, stream=stream, timeout=30)
        if resp.status_code != 200:
            print(f"[本地模型] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        if stream:
            return resp.iter_lines()
        else:
            return resp.json().get("response", "")
    except Exception as e:
        print(f"[本地模型] 请求异常: {e}")
        return None

# ---------- 工具判断提示词 ----------
TOOL_DETECTION_PROMPT = """你是一个工具调用助手。根据用户的问题，判断是否需要使用以下工具：
- get_current_time: 获取当前系统时间
- get_all_users: 获取数据库中的用户列表
- run_python_code: 执行一段 Python 代码
- web_search: 在互联网上搜索信息
- read_file: 读取文件内容
- write_file: 写入内容到文件
- get_weather: 查询城市实时天气

**重要**：如果需要工具，你必须**只输出**工具名，例如 "get_current_time"，不要输出任何其他文字。
如果不需要工具，只输出 "NO_TOOL"。

用户问题：{}
"""

# ---------- 云端模型配置（备选）----------
MODELS_CONFIG = [
    {"name": "Qwen3-8B", "model_id": "Qwen/Qwen3-8B", "note": "主力"},
    {"name": "DeepSeek-R1-Distill-Qwen-7B", "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "note": "备选"}
]
SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY')
if not SILICONFLOW_API_KEY:
    print("[警告] 未设置 SILICONFLOW_API_KEY，仅使用本地模型")

# ---------- 数据库配置（直接运行Flask时，MySQL通常在localhost）----------
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', '123456'),
    'database': os.environ.get('MYSQL_DATABASE', 'myapp'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

# ---------- 工具函数 ----------
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_all_users():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, email FROM users")
            users = cursor.fetchall()
        return users
    finally:
        conn.close()

def web_search(query: str, max_results=3) -> str:
    print(f"[SEARCH] 收到搜索请求，关键词: {query}", flush=True)
    try:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = soup.select('.result')
        if not results:
            return "未找到相关结果。"
        output = []
        for i, res in enumerate(results[:max_results]):
            title_elem = res.select_one('.result__title')
            link_elem = res.select_one('.result__url')
            snippet_elem = res.select_one('.result__snippet')
            title = title_elem.get_text(strip=True) if title_elem else "无标题"
            link = link_elem.get('href') if link_elem else ""
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            output.append(f"{i+1}. {title}\n   {snippet}\n   链接: {link}")
        return "\n\n".join(output)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"搜索失败: {str(e)}"

def run_python_code(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    try:
        result = subprocess.run(['python3', temp_file], capture_output=True, text=True, timeout=10)
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or "[无输出]"
    except subprocess.TimeoutExpired:
        return "执行超时（>10秒）"
    finally:
        os.unlink(temp_file)

def read_file(filepath: str) -> str:
    """读取文件内容（安全限制在当前目录）"""
    base = os.path.abspath('.')
    full = os.path.abspath(os.path.join(base, filepath))
    if not full.startswith(base):
        return "错误：不允许访问上级目录。"
    try:
        with open(full, 'r', encoding='utf-8') as f:
            return f.read()[:2000]
    except Exception as e:
        return f"读取失败: {e}"

def write_file(filepath: str, content: str) -> str:
    """写入内容到文件（覆盖模式）"""
    base = os.path.abspath('.')
    full = os.path.abspath(os.path.join(base, filepath))
    if not full.startswith(base):
        return "错误：不允许写入上级目录。"
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入 {len(content)} 字符到 {filepath}"
    except Exception as e:
        return f"写入失败: {e}"

# 在第 120 行左右（write_file 函数后面）添加：
def get_weather(city: str) -> str:
    """使用高德地图 API 获取实时天气"""
    api_key = os.environ.get('AMAP_API_KEY')
    if not api_key:
        return "未配置高德天气 API Key，请在环境变量中设置 AMAP_API_KEY"
    try:
        # 1. 先根据城市名获取城市编码（adcode）
        geocode_url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&output=json&key={api_key}"
        geo_resp = requests.get(geocode_url, timeout=5)
        geo_data = geo_resp.json()
        if geo_data['status'] != '1' or not geo_data['geocodes']:
            return f"未找到城市 '{city}'"
        adcode = geo_data['geocodes'][0]['adcode']
        # 2. 根据城市编码获取天气
        weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&output=json&key={api_key}"
        weather_resp = requests.get(weather_url, timeout=5)
        weather_data = weather_resp.json()
        if weather_data['status'] != '1' or not weather_data['lives']:
            return f"无法获取 '{city}' 的天气"
        live = weather_data['lives'][0]
        return (f"{live['city']}：{live['weather']}，"
                f"温度 {live['temperature']}℃，"
                f"湿度 {live['humidity']}%，"
                f"风向 {live['winddirection']}，风力 {live['windpower']} 级")
    except Exception as e:
        return f"天气查询失败: {str(e)}"

TOOLS_LIST = {
    "get_current_time": get_current_time,
    "get_all_users": get_all_users,
    "run_python_code": run_python_code,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "get_weather": get_weather   # 添加这一行
}

# 关键词兜底映射（当模型判断失败时使用）
TOOL_KEYWORDS_MAP = {
    'get_current_time': ['现在几点', '当前时间', '几点了', '什么时间', '几时'],
    'get_all_users': ['列出用户', '所有用户', '用户列表', '有哪些用户', '显示用户'],
    'run_python_code': ['计算', '运行代码', '执行python', '写代码', '代码'],
    'web_search': ['搜索', '查找', '网上搜一下', '百度一下'],
    'read_file': ['读取文件', '查看文件', '打开文件'],
    'write_file': ['写入文件', '保存到文件', '创建文件'],
    'get_weather': ['天气', '气温', '会不会下雨', '今天天气', '天气预报']
}

SYSTEM_PROMPT = """你是江洋大模型。回答时请始终以“我”的第一人称角度，语言自然、客观、详细。
如果用户询问工具执行的结果，请基于提供的工具结果来回答。"""

def json_response(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False, indent=2), status=status, mimetype='application/json')

# ---------- 云端模型链（备选）----------
def call_cloud_llm(messages, stream=False, tools=None, tool_choice="auto"):
    if not SILICONFLOW_API_KEY:
        return None, None
    for model_conf in MODELS_CONFIG:
        model_id = model_conf["model_id"]
        try:
            print(f"[云端] 尝试调用: {model_id}")
            client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1", timeout=20.0)
            params = {"model": model_id, "messages": messages, "temperature": 0.7}
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            if stream:
                params["stream"] = True
                return client.chat.completions.create(**params), model_id
            else:
                resp = client.chat.completions.create(**params)
                if resp.choices and resp.choices[0].message.content:
                    return resp, model_id
        except Exception as e:
            print(f"[云端] {model_id} 失败: {e}")
            continue
    return None, None

def generate_code_for_query(query: str) -> str:
    """让本地模型生成解决用户问题的 Python 代码"""
    code_prompt = f"请生成 Python 代码来解决以下问题，只输出代码，不要添加解释：{query}"
    code = call_local_llm(code_prompt, system_prompt="", stream=False, max_tokens=256)
    if code and "print" not in code and "=" not in code:
        # 如果返回的是纯数字，包装成 print
        if code.strip().isdigit():
            code = f"print({code})"
    return code

def extract_search_query(user_query: str) -> str:
    """从用户问题中提取搜索关键词"""
    # 简单正则：匹配“搜索xxx”或“查找xxx”
    match = re.search(r'搜索[\s：:]*([^。]+)', user_query)
    if match:
        return match.group(1).strip()
    match = re.search(r'查找[\s：:]*([^。]+)', user_query)
    if match:
        return match.group(1).strip()
    # 如果用户直接问“xx是什么”，也可以当作搜索词
    return user_query.strip()

def extract_filepath(user_query: str) -> str:
    """从用户问题中提取文件路径"""
    match = re.search(r'读取文件[\s]*([^\s]+)', user_query)
    if match:
        return match.group(1)
    match = re.search(r'文件[\s]*([^\s]+)', user_query)
    if match:
        return match.group(1)
    return None

def extract_write_content(user_query: str):
    """提取写入文件的路径和内容，格式：写入文件 path 内容为 content"""
    match = re.search(r'写入文件[\s]*([^\s]+)[\s]*内容为[\s]*(.+)', user_query)
    if match:
        return match.group(1), match.group(2)
    # 另一种格式：保存到文件 path ： content
    match = re.search(r'保存到文件[\s]*([^\s]+)[\s]*:[\s]*(.+)', user_query)
    if match:
        return match.group(1), match.group(2)
    return None, None

# ---------- 主聊天路由 ----------
@app.route('/chat-stream', methods=['GET'])
def chat_stream():
    user_query = request.args.get('q', '')
    if not user_query:
        return json_response({'error': '请提供参数 q'}, 400)

    # 获取或生成 session_id（用于区分用户）
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())

    def generate():
        yield ANSWER_PREFIX + "\n"

        # 从 Redis 加载历史对话（最多最近 20 条消息）
        history_key = f"chat_history:{session_id}"
        history_raw = redis_client.lrange(history_key, -20, -1) if redis_client else []
        history_messages = []
        for item in history_raw:
            try:
                history_messages.append(json.loads(item))
            except:
                pass

        # 构建消息列表：系统提示 + 历史 + 当前用户问题
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_query})

        # ---------- 第一阶段：智能工具判断（本地模型）----------
        tool_name = None
        try:
            detection_prompt = TOOL_DETECTION_PROMPT.format(user_query)
            detection_response = call_local_llm(detection_prompt, system_prompt="", stream=False, max_tokens=20)
            if detection_response and detection_response != "NO_TOOL":
                candidate = detection_response.strip()
                if candidate in TOOLS_LIST:
                    tool_name = candidate
                    print(f"[工具判断] 模型决定调用工具: {tool_name}")
        except Exception as e:
            print(f"[工具判断] 模型调用失败: {e}")

        # ---------- 第二阶段：关键词兜底 ----------
        if not tool_name:
            for tool, keywords in TOOL_KEYWORDS_MAP.items():
                if any(kw in user_query for kw in keywords):
                    tool_name = tool
                    print(f"[工具判断] 关键词触发工具: {tool_name}")
                    break

        # ---------- 第三阶段：执行工具 ----------
        tool_result = None
        if tool_name and tool_name in TOOLS_LIST:
            try:
                if tool_name == "run_python_code":
                    code = generate_code_for_query(user_query)
                    if code:
                        print(f"[代码生成] {code}")
                        tool_result = run_python_code(code)
                    else:
                        tool_result = "无法生成有效的 Python 代码。"
                elif tool_name == "web_search":
                    query = extract_search_query(user_query)
                    tool_result = web_search(query)
                elif tool_name == "read_file":
                    filepath = extract_filepath(user_query)
                    if filepath:
                        tool_result = read_file(filepath)
                    else:
                        tool_result = "请指定要读取的文件名，例如：读取文件 test.txt"
                elif tool_name == "write_file":
                    path, content = extract_write_content(user_query)
                    if path and content:
                        tool_result = write_file(path, content)
                    else:
                        tool_result = "请使用格式：写入文件 文件名 内容为 ... 或 保存到文件 path : content"
                elif tool_name == "get_weather":
                    # 提取城市名（简单实现：取“天气”后面的词）
                    import re
                    match = re.search(r'([^天气]+)天气', user_query)
                    city = match.group(1).strip() if match else "北京"
                    tool_result = get_weather(city) 
                
                else:
                    tool_result = TOOLS_LIST[tool_name]()
                # 将工具结果作为用户消息追加，便于模型参考
                messages.append({"role": "user", "content": f"工具 '{tool_name}' 执行结果：{tool_result}"})
                yield f"[工具结果] {tool_result}\n"
            except Exception as e:
                error_msg = f"工具执行失败: {str(e)}"
                messages.append({"role": "user", "content": error_msg})
                yield f"[工具错误] {error_msg}\n"

        # ---------- 第四阶段：生成最终回答 ----------
        # 构造对话文本（因为本地 Ollama 不支持 messages 格式，需拼成自然文本）
        conversation_text = ""
        for msg in messages:
            if msg["role"] == "system":
                conversation_text += f"系统：{msg['content']}\n"
            elif msg["role"] == "user":
                conversation_text += f"用户：{msg['content']}\n"
        conversation_text += "助手："

        final_stream = call_local_llm(conversation_text, system_prompt="", stream=True, max_tokens=1024)
        full_answer = []
        if final_stream:
            for line in final_stream:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        token = data.get("response", "")
                        if token:
                            full_answer.append(token)
                            yield token
                    except:
                        pass
            # 将本次对话存储到 Redis
            assistant_message = {"role": "assistant", "content": "".join(full_answer)}
            user_message = {"role": "user", "content": user_query}
            if redis_client:
                redis_client.rpush(history_key, json.dumps(user_message))
                redis_client.rpush(history_key, json.dumps(assistant_message))
                redis_client.expire(history_key, 604800)  # 7天过期
            return

        # 本地模型失败，尝试云端
        cloud_resp, used_model = call_cloud_llm(messages, stream=True)
        if cloud_resp:
            for chunk in cloud_resp:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer.append(token)
                    yield token
            # 存储到 Redis
            assistant_message = {"role": "assistant", "content": "".join(full_answer)}
            user_message = {"role": "user", "content": user_query}
            if redis_client:
                redis_client.rpush(history_key, json.dumps(user_message))
                redis_client.rpush(history_key, json.dumps(assistant_message))
                redis_client.expire(history_key, 604800)
            print(f"[完成] 使用云端模型: {used_model}")
        else:
            yield "\n[错误] 所有模型均不可用，请检查 Ollama 服务或云端配置。"

    response = Response(stream_with_context(generate()), mimetype='text/plain; charset=utf-8')
    response.set_cookie('session_id', session_id, max_age=604800, httponly=True)
    return response

# ---------- 辅助路由 ----------
@app.route('/')
def hello():
    return f'<h1>{MODEL_DISPLAY_NAME}</h1><p><a href="/chat-ui">进入对话</a></p>'

@app.route('/chat-ui')
def chat_ui():
    return render_template('chat.html')

@app.route('/users')
def list_users():
    return json_response(get_all_users())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)