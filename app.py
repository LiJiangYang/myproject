import os
import datetime
import subprocess
import tempfile
import json
import sys
import time
import re
import uuid
import logging
from logging.handlers import RotatingFileHandler

import requests
from flask import Flask, request, Response, stream_with_context, render_template, make_response
import pymysql
from openai import OpenAI
from bs4 import BeautifulSoup
import redis

app = Flask(__name__)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------- 日志配置 ----------------------------
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not app.debug:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10240,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    app.logger.addHandler(stream_handler)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

MODEL_DISPLAY_NAME = "江洋大模型（联网搜索版）"
ANSWER_PREFIX = f"【{MODEL_DISPLAY_NAME}】"

# ---------------------------- 本地 Ollama ----------------------------
LOCAL_MODEL_NAME = "qwen2:1.5b-instruct"
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'localhost')
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:11434/api/generate"

# ---------------------------- Redis ----------------------------
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        decode_responses=True
    )
    redis_client.ping()
    app.logger.info("Redis 连接成功")
except Exception as e:
    app.logger.warning(f"Redis 连接失败，将不使用记忆功能: {e}")

# ---------------------------- 本地模型调用 ----------------------------
def call_local_llm(prompt, system_prompt="", stream=False, max_tokens=512):
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
            app.logger.error(f"本地模型 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        if stream:
            return resp.iter_lines()
        else:
            return resp.json().get("response", "")
    except Exception as e:
        app.logger.error(f"本地模型请求异常: {e}")
        return None

# ---------------------------- 工具判断提示词 ----------------------------
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

# ---------------------------- 云端模型备选 ----------------------------
MODELS_CONFIG = [
    {"name": "Qwen3-8B", "model_id": "Qwen/Qwen3-8B"},
    {"name": "DeepSeek-R1-Distill-Qwen-7B", "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
]
SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY')
if not SILICONFLOW_API_KEY:
    app.logger.warning("未设置 SILICONFLOW_API_KEY，仅使用本地模型")

def call_cloud_llm(messages, stream=False, tools=None, tool_choice="auto"):
    if not SILICONFLOW_API_KEY:
        return None, None
    for model_conf in MODELS_CONFIG:
        model_id = model_conf["model_id"]
        try:
            app.logger.info(f"云端尝试: {model_id}")
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
            app.logger.error(f"云端模型 {model_id} 失败: {e}")
            continue
    return None, None

# ---------------------------- 数据库 ----------------------------
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

# ---------------------------- 工具函数 ----------------------------
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_all_users():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, email FROM users")
            return cursor.fetchall()
    finally:
        conn.close()

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

def web_search(query: str, max_results=3) -> str:
    """博查 API 搜索，带缓存，符合官方文档"""
    if not hasattr(web_search, 'cache'):
        web_search.cache = {}
    now = time.time()
    cache_key = query
    if cache_key in web_search.cache:
        res, exp = web_search.cache[cache_key]
        if now < exp:
            app.logger.info(f"[搜索缓存命中] 关键词: {query}")
            return res
        else:
            del web_search.cache[cache_key]

    api_key = os.environ.get('BOCHA_API_KEY')
    if not api_key:
        err = "未配置 BOCHA_API_KEY 环境变量"
        app.logger.error(err)
        return err

    url = "https://api.bocha.cn/v1/web-search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # 根据文档，使用 summary 和 count 参数
    payload = {
        "query": query,
        "summary": True,
        "count": max_results
    }
    app.logger.info(f"[博查搜索] 请求: {query}, count={max_results}")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        app.logger.info(f"[博查搜索] 响应状态码: {resp.status_code}")
        if resp.status_code != 200:
            app.logger.error(f"[博查搜索] HTTP 错误: {resp.status_code}, 响应: {resp.text[:500]}")
            return f"搜索失败: HTTP {resp.status_code}"
        data = resp.json()
        app.logger.info(f"[博查搜索] 响应体预览: {json.dumps(data, ensure_ascii=False)[:500]}")
        
        # 检查 code 字段
        if data.get('code') != 200:
            err_msg = data.get('msg', '未知错误')
            app.logger.error(f"[博查搜索] API 返回错误: {err_msg}")
            return f"搜索失败: {err_msg}"
        
        # 官方结构: data.data.webPages.value
        web_pages = data.get('data', {}).get('webPages', {})
        results = web_pages.get('value', [])
        
        if not results:
            result_str = "未找到相关结果。"
        else:
            output = []
            for i, item in enumerate(results[:max_results], 1):
                title = item.get('name', '无标题')
                # snippet 和 summary 二选一
                snippet = item.get('snippet') or item.get('summary', '')
                output.append(f"{i}. {title}\n   {snippet}")
            result_str = "\n\n".join(output)
        
        # 缓存
        web_search.cache[cache_key] = (result_str, now + 600)
        if len(web_search.cache) > 20:
            oldest = min(web_search.cache.items(), key=lambda x: x[1][1])[0]
            del web_search.cache[oldest]
        app.logger.info(f"[博查搜索] 成功返回 {len(results)} 条结果")
        return result_str
    except requests.exceptions.Timeout:
        app.logger.error("[博查搜索] 请求超时")
        return "搜索超时，请稍后再试。"
    except requests.exceptions.RequestException as e:
        app.logger.error(f"[博查搜索] 网络异常: {e}")
        return f"搜索请求失败: {str(e)}"
    except Exception as e:
        app.logger.error(f"[博查搜索] 未知异常: {e}", exc_info=True)
        return f"搜索失败: {str(e)}"
    
def read_file(filepath: str) -> str:
    base = os.path.abspath('.')
    full = os.path.abspath(os.path.join(base, filepath))
    if not full.startswith(base):
        return "不允许访问上级目录"
    try:
        with open(full, 'r', encoding='utf-8') as f:
            return f.read()[:2000]
    except Exception as e:
        return f"读取失败: {e}"

def write_file(filepath: str, content: str) -> str:
    base = os.path.abspath('.')
    full = os.path.abspath(os.path.join(base, filepath))
    if not full.startswith(base):
        return "不允许写入上级目录"
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入 {len(content)} 字符"
    except Exception as e:
        return f"写入失败: {e}"

def get_weather(city: str) -> str:
    api_key = os.environ.get('AMAP_API_KEY')
    if not api_key:
        return "未配置高德天气 API Key"
    try:
        geocode_url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&output=json&key={api_key}"
        geo_resp = requests.get(geocode_url, timeout=5)
        geo_data = geo_resp.json()
        if geo_data['status'] != '1' or not geo_data['geocodes']:
            return f"未找到城市 {city}"
        adcode = geo_data['geocodes'][0]['adcode']
        weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&output=json&key={api_key}"
        weather_resp = requests.get(weather_url, timeout=5)
        weather_data = weather_resp.json()
        if weather_data['status'] != '1' or not weather_data['lives']:
            return f"无法获取 {city} 天气"
        live = weather_data['lives'][0]
        return f"{live['city']}：{live['weather']}，温度 {live['temperature']}℃，湿度 {live['humidity']}%，风向 {live['winddirection']}，风力 {live['windpower']} 级"
    except Exception as e:
        return f"天气查询失败: {e}"

TOOLS_LIST = {
    "get_current_time": get_current_time,
    "get_all_users": get_all_users,
    "run_python_code": run_python_code,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "get_weather": get_weather
}

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

def generate_code_for_query(query: str) -> str:
    code_prompt = f"请生成 Python 代码来解决以下问题，只输出代码，不要添加解释：{query}"
    code = call_local_llm(code_prompt, system_prompt="", stream=False, max_tokens=256)
    if code and "print" not in code and "=" not in code:
        if code.strip().isdigit():
            code = f"print({code})"
    return code

def extract_search_query(user_query: str) -> str:
    match = re.search(r'搜索[\s：:]*([^。]+)', user_query)
    if match:
        return match.group(1).strip()
    match = re.search(r'查找[\s：:]*([^。]+)', user_query)
    if match:
        return match.group(1).strip()
    return user_query.strip()

def extract_filepath(user_query: str) -> str:
    match = re.search(r'读取文件[\s]*([^\s]+)', user_query)
    if match:
        return match.group(1)
    match = re.search(r'文件[\s]*([^\s]+)', user_query)
    if match:
        return match.group(1)
    return None

def extract_write_content(user_query: str):
    match = re.search(r'写入文件[\s]*([^\s]+)[\s]*内容为[\s]*(.+)', user_query)
    if match:
        return match.group(1), match.group(2)
    match = re.search(r'保存到文件[\s]*([^\s]+)[\s]*:[\s]*(.+)', user_query)
    if match:
        return match.group(1), match.group(2)
    return None, None

# ---------------------------- 主路由 ----------------------------
@app.route('/chat-stream', methods=['GET'])
def chat_stream():
    user_query = request.args.get('q', '')
    if not user_query:
        return json_response({'error': '请提供参数 q'}, 400)

    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())

    app.logger.info(f"[请求] session={session_id[:8]}, query={user_query}")

    def generate():
        yield ANSWER_PREFIX + "\n"

        # 加载历史
        history_key = f"chat_history:{session_id}"
        history_messages = []
        if redis_client:
            try:
                raw = redis_client.lrange(history_key, -20, -1)
                for item in raw:
                    history_messages.append(json.loads(item))
                app.logger.debug(f"加载 {len(history_messages)} 条历史消息")
            except Exception as e:
                app.logger.error(f"Redis 读取失败: {e}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_query})

        # ---------- 工具判断 ----------
        tool_name = None
        try:
            detection_prompt = TOOL_DETECTION_PROMPT.format(user_query)
            detection_response = call_local_llm(detection_prompt, system_prompt="", stream=False, max_tokens=20)
            if detection_response and detection_response != "NO_TOOL":
                candidate = detection_response.strip()
                if candidate in TOOLS_LIST:
                    tool_name = candidate
                    app.logger.info(f"[工具判断] 模型决定调用: {tool_name}")
                else:
                    app.logger.warning(f"[工具判断] 模型返回未知工具: {candidate}")
        except Exception as e:
            app.logger.error(f"[工具判断] 模型调用失败: {e}")

        if not tool_name:
            for tool, keywords in TOOL_KEYWORDS_MAP.items():
                if any(kw in user_query for kw in keywords):
                    tool_name = tool
                    app.logger.info(f"[工具判断] 关键词触发: {tool_name}")
                    break

        # ---------- 执行工具 ----------
        tool_result = None
        if tool_name and tool_name in TOOLS_LIST:
            app.logger.info(f"[工具执行] 开始调用 {tool_name}")
            try:
                if tool_name == "run_python_code":
                    code = generate_code_for_query(user_query)
                    if code:
                        app.logger.debug(f"[代码生成] 代码: {code[:200]}")
                        tool_result = run_python_code(code)
                    else:
                        tool_result = "无法生成有效的 Python 代码"
                elif tool_name == "web_search":
                    query = extract_search_query(user_query)
                    app.logger.info(f"[搜索] 关键词: {query}")
                    tool_result = web_search(query)
                elif tool_name == "read_file":
                    fp = extract_filepath(user_query)
                    if fp:
                        tool_result = read_file(fp)
                    else:
                        tool_result = "请指定要读取的文件名"
                elif tool_name == "write_file":
                    path, content = extract_write_content(user_query)
                    if path and content:
                        tool_result = write_file(path, content)
                    else:
                        tool_result = "请使用格式：写入文件 文件名 内容为 ..."
                elif tool_name == "get_weather":
                    match = re.search(r'([^天气]+)天气', user_query)
                    city = match.group(1).strip() if match else "北京"
                    app.logger.info(f"[天气] 查询城市: {city}")
                    tool_result = get_weather(city)
                else:
                    tool_result = TOOLS_LIST[tool_name]()
                messages.append({"role": "user", "content": f"工具 '{tool_name}' 执行结果：{tool_result}"})
                app.logger.info(f"[工具执行] {tool_name} 完成，结果长度: {len(str(tool_result))}")
            except Exception as e:
                error_msg = f"工具执行失败: {str(e)}"
                messages.append({"role": "user", "content": error_msg})
                app.logger.error(f"[工具执行] {tool_name} 异常: {e}", exc_info=True)

        # ---------- 知识性问题路由 ----------
        knowledge_keywords = ['是什么', '什么是', '谁', '为什么', '介绍', '定义', '解释', '历史', '原理', '区别', '知道', '了解', '能否介绍', '说说']
        is_knowledge = any(kw in user_query for kw in knowledge_keywords)
        if is_knowledge and '1.5b' in LOCAL_MODEL_NAME and not tool_name:
            app.logger.info("[路由] 知识性问题，使用云端模型")
            cloud_resp, used = call_cloud_llm(messages, stream=True)
            if cloud_resp:
                full = []
                for chunk in cloud_resp:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full.append(token)
                        yield token
                if redis_client:
                    redis_client.rpush(history_key, json.dumps({"role": "user", "content": user_query}))
                    redis_client.rpush(history_key, json.dumps({"role": "assistant", "content": "".join(full)}))
                    redis_client.expire(history_key, 604800)
                app.logger.info(f"[完成] 云端模型回答长度: {len(''.join(full))}")
                return

        # ---------- 工具后优先云端 ----------
        if tool_name:
            app.logger.info("[路由] 工具已使用，调用云端模型生成最终回答")
            cloud_resp, used = call_cloud_llm(messages, stream=True)
            if cloud_resp:
                full = []
                for chunk in cloud_resp:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full.append(token)
                        yield token
                if redis_client:
                    redis_client.rpush(history_key, json.dumps({"role": "user", "content": user_query}))
                    redis_client.rpush(history_key, json.dumps({"role": "assistant", "content": "".join(full)}))
                    redis_client.expire(history_key, 604800)
                app.logger.info(f"[完成] 云端模型回答长度: {len(''.join(full))}")
                return

        # ---------- 本地模型 ----------
        app.logger.info("[路由] 使用本地模型生成回答")
        conv_text = ""
        for msg in messages:
            if msg["role"] == "system":
                conv_text += f"系统：{msg['content']}\n"
            elif msg["role"] == "user":
                conv_text += f"用户：{msg['content']}\n"
        conv_text += "助手："
        stream_gen = call_local_llm(conv_text, system_prompt="", stream=True, max_tokens=1024)
        if stream_gen:
            full = []
            for line in stream_gen:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        token = data.get("response", "")
                        if token:
                            full.append(token)
                            yield token
                    except:
                        pass
            if redis_client:
                redis_client.rpush(history_key, json.dumps({"role": "user", "content": user_query}))
                redis_client.rpush(history_key, json.dumps({"role": "assistant", "content": "".join(full)}))
                redis_client.expire(history_key, 604800)
            app.logger.info(f"[完成] 本地模型回答长度: {len(''.join(full))}")
            return

        # 最终降级
        err_msg = "\n[错误] 无法生成回答，请稍后重试。"
        yield err_msg
        app.logger.error("所有模型均不可用")

    response = Response(stream_with_context(generate()), mimetype='text/plain; charset=utf-8')
    response.set_cookie('session_id', session_id, max_age=604800, httponly=True)
    return response

# ---------------------------- 辅助路由 ----------------------------
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