from utils import Ai
from utils import file
import base64
import json
import os
import requests
import logging
from openai import OpenAI
import re  
import threading
import concurrent.futures


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return config_data

def clean_path(path):
    """
    检查并移除路径字符串中的引号。
    Args:
        path (str): 原始路径字符串。
    Returns:
        str: 清理后的路径字符串。
    """
    if path.startswith("'") and path.endswith("'"):
        path = path[1:-1]
    elif path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    return path

def get_image_paths():
    image_extensions = ('.jpg', '.jpeg', '.png')
    current_dir = os.getcwd()
    image_paths = []
    for root, dirs, files in os.walk(current_dir):
        if 'log' in dirs:
            dirs.remove('log')
        if os.path.basename(root) == 'AiBioNoteGen': 
            pass
            
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return image_paths

def del_images(images):
    deleted_count = 0  
    for image in images:
        try:
            os.remove(image)
            logging.info(f"已删除: {image}")
            deleted_count += 1
        except Exception as e:
            logging.error(f"删除失败 {image}: {str(e)}")
    logging.info(f"共删除了 {deleted_count} 张图片。")

def save_files_from_json(json_string, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        json_string = re.sub(r'^```json\s*|\s*```$', '', json_string, flags=re.MULTILINE)
        files_data = json.loads(json_string)
        if not isinstance(files_data, list):
            logging.error("JSON顶层结构不是一个列表，无法处理。")
            return

        for file_info in files_data:
            filename = file_info.get('filename')
            content = file_info.get('content')
            if not filename or content is None:
                logging.warning(f"跳过无效的文件数据: {file_info}")
                continue

            file_path = os.path.join(output_dir, filename)
            if '..' in filename or os.path.isabs(filename):
                logging.error(f"检测到不安全的文件名，已跳过: {filename}")
                continue

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"成功创建文件: {file_path}")

    except json.JSONDecodeError as e:
        logging.error(f"JSON解析失败: {e}")
        logging.error(f"原始响应内容:\n---\n{json_string}\n---")
    except Exception as e:
        logging.error(f"创建文件时发生未知错误: {e}")

def prompt_reader(name:str) ->str:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_path, name), 'r', encoding='utf-8') as f:
        prompt = f.read()
    return prompt

def save_files_from_response(response_text: str, output_dir: str):
    """
    解析由最终模型生成的、包含多个文件的单一文本响应，并保存它们。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    file_separator = "###-###-END-OF-FILE-###-###"
    file_blocks = response_text.strip().split(file_separator)
    
    saved_count = 0
    for block in file_blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split('\n', 1)
        if not lines or not lines[0].startswith('FILENAME:'):
            logging.warning(f"跳过格式不正确的文本块: {block[:100]}...")
            continue
        filename = lines[0].replace('FILENAME:', '').strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        if not filename or '..' in filename or os.path.isabs(filename):
            logging.error(f"检测到不安全或无效的文件名，已跳过: {filename}")
            continue

        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"成功创建文件: {file_path}")
            saved_count += 1
        except Exception as e:
            logging.error(f"创建文件 {filename} 时发生错误: {e}")
            
    logging.info(f"从单次响应中总共保存了 {saved_count} 个文件。")

def run_first_draft_generation(image_path, vision_agent, vision_prompt):
    """
    工作单元函数，仅执行Stage 1：根据图片生成初稿。
    被线程池中的每个线程调用。
    """
    thread_name = threading.current_thread().name
    image_name = os.path.basename(image_path)
    logging.info(f"[{thread_name}] 开始生成初稿: {image_name}")
    print(f"[{thread_name}] 开始生成初稿: {image_name}")
    
    try:
        image_encoded = [Ai.encode_image(image_path)]
        first_draft = vision_agent.chat(vision_prompt, image_encoded, False)
        logging.info(f"[{thread_name}] 成功为 {image_name} 生成初稿。")
        print(f"[{thread_name}] 成功为 {image_name} 生成初稿。")
        return first_draft
    except Exception as e:
        logging.error(f"[{thread_name}] 生成初稿失败 for {image_name}: {e}")
        print(f"[{thread_name}] 生成初稿失败 for {image_name}: {e}")
        return ""

def v050():
    # 同时运行的vision数量限制
    concurrency = 10 
    
    # --- 1. 配置与初始化 ---
    logging.info("--- 初始化配置与AI Agent ---")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    conf = load_config(os.path.join(cur_path, 'key-api.json'))
    
    # 读取主模板和三个阶段的包装Prompt
    master_prompt = prompt_reader("master_prompt.txt")
    vision_prompt_template = prompt_reader("vision.txt")
    build_prompt_template = prompt_reader("build.txt")
    gen_prompt_template = prompt_reader("gen.txt")
    
    # 构造完整的、可直接使用的Prompt
    vision_prompt = vision_prompt_template.replace("[此处由程序粘贴 master_prompt.txt 的全部内容]", master_prompt)
    
    # Agent的角色现在都统一为知识库架构师，因为它们都遵循同一个主模板
    vision_agent = Ai.LLM(conf["vision"]["api_key"], conf["vision"]["base_url"], conf["vision"]["model"], "你是一位知识库架构师大师")
    build_agent = Ai.LLM(conf["review"]["api_key"], conf["review"]["base_url"], conf["review"]["model"], "你是一位知识库架构师大师")
    gen_agent = Ai.LLM(conf["formatting"]["api_key"], conf["formatting"]["base_url"], conf["formatting"]["model"], "你是一位知识库架构师大师")

    output_directory = os.path.join(cur_path, 'Obsidian-Notes')

    # --- 2. 扇出 (Fan-out): 并行生成所有图片的初稿 ---
    image_paths = get_image_paths()
    if not image_paths:
        logging.warning("未找到任何图片，程序退出。")
        return
        
    logging.info(f"--- 阶段1 (并行): 发现 {len(image_paths)} 张图片，启动最多{concurrency}个线程生成初稿 ---")
    
    all_first_drafts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_image = {executor.submit(run_first_draft_generation, img_path, vision_agent, vision_prompt): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(future_to_image):
            try:
                result_draft = future.result()
                if result_draft and result_draft.strip():
                    all_first_drafts.append(result_draft)
            except Exception as exc:
                logging.error(f"处理图片 {os.path.basename(future_to_image[future])} 的结果时产生异常: {exc}")

    if not all_first_drafts:
        logging.warning("所有图片均未能生成有效初稿，程序终止。")
        del_images(image_paths) 
        return
        
    # --- 3. 聚合与迭代优化 ---
    # 将所有初稿聚合为一个大文本块
    aggregated_draft = "\n".join(all_first_drafts)
    
    # --- STAGE 2: Build - 结构与内容优化 ---
    logging.info("--- 阶段2 (顺序): 开始对聚合后的草稿进行结构与内容优化 ---")
    try:
        build_prompt = build_prompt_template.replace("[此处由程序粘贴 Vision 阶段生成的草稿]", aggregated_draft)
        build_prompt = build_prompt.replace("[此处由程序粘贴 master_prompt.txt 的全部内容]", master_prompt)
        refined_draft = build_agent.chat(build_prompt)
        logging.info("结构与内容优化完成。")
    except Exception as e:
        logging.error(f"阶段2 Build 失败: {e}")
        refined_draft = aggregated_draft # 如果Build失败，就用原始草稿进行下一步

    # --- STAGE 3: Gen - 最终格式化与渲染 ---
    logging.info("--- 阶段3 (顺序): 开始进行最终的格式化与链接渲染 ---")
    try:
        gen_prompt = gen_prompt_template.replace("[此处由程序粘贴 Build 阶段生成的草稿]", refined_draft)
        gen_prompt = gen_prompt.replace("[此处由程序粘贴 master_prompt.txt 的全部内容]", master_prompt)
        final_output = gen_agent.chat(gen_prompt)
        logging.info("最终渲染完成。")
    except Exception as e:
        logging.error(f"阶段3 Gen 失败: {e}")
        final_output = refined_draft # 如果Gen失败，就使用Build的结果

    # --- 4. 文件写入与清理 ---
    logging.info("--- 开始解析并写入最终文件 ---")
    save_files_from_response(final_output, output_directory)
    
    logging.info("--- 清理已处理的图片 ---")
    del_images(image_paths)
    
    logging.info("--- 所有任务完成 ---")

# 在主执行块中调用新函数
if __name__ == "__main__":
    # log
    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(current_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'main.log')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    # 主要部分
    v050()
