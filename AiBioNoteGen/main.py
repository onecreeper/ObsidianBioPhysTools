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


def run_vision_stage_for_image(image_path, vision_agent, prompt_stage1):
    """
    工作单元函数，仅执行Stage 1：视觉提取。
    它被线程池中的每个线程调用，并返回一个包含概念和上下文的Markdown格式字符串。
    """
    thread_name = threading.current_thread().name
    image_name = os.path.basename(image_path)
    logging.info(f"[{thread_name}] 开始视觉提取: {image_name}")
    print(f"[{thread_name}] 开始视觉提取: {image_name}") # 保留print以提供即时反馈
    
    try:
        image_encoded = [Ai.encode_image(image_path)]
        # LLM现在被要求输出Markdown格式的文本，而不是JSON
        markdown_output = vision_agent.chat(prompt_stage1, image_encoded, False)
        logging.info(f"[{thread_name}] 成功为 {image_name} 提取到Markdown上下文。")
        print(f"[{thread_name}] 成功为 {image_name} 提取到Markdown上下文。")
        # 直接返回原始的Markdown字符串
        return markdown_output
    except Exception as e:
        logging.error(f"[{thread_name}] 视觉提取失败 for {image_name}: {e}")
        print(f"[{thread_name}] 视觉提取失败 for {image_name}: {e}")
        return "" # 失败时返回空字符串，避免中断整个流程

    
def v01():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    conf = load_config(os.path.join(cur_path, 'key-api.json'))
    
    # === 阶段1：视觉识别与草稿生成 ===
    logging.info("--- 阶段1：视觉识别与草稿生成 ---")
    image_paths = get_image_paths()
    if not image_paths:
        logging.warning("未找到任何图片，程序退出。")
        exit()
        
    images_encoded = []
    for image_path in image_paths: 
        images_encoded.append(Ai.encode_image(image_path))
        
    with open(os.path.join(cur_path, 'vision.txt'), 'r', encoding='utf-8') as f:
        vision_prompt = f.read()

    agent1_name = "vision" 
    vision_agent = Ai.LLM(conf[agent1_name]["api_key"], conf[agent1_name]["base_url"], conf[agent1_name]["model"], "你是一名高中生物老师")
    draft_res = vision_agent.chat(vision_prompt, images_encoded) 
    logging.info("阶段1完成，已生成草稿。")

    # === 阶段2：校对与修正 ===
    logging.info("--- 阶段2：校对与修正 ---")
    with open(os.path.join(cur_path, 'review.txt'), 'r', encoding='utf-8') as f:
        review_prompt = f.read()
        
    agent2_name = "review" 
    review_agent = Ai.LLM(conf[agent2_name]["api_key"], conf[agent2_name]["base_url"], conf[agent2_name]["model"], "你是一名拥有绝对权限的终极知识库校对官")
    corrected_res = review_agent.chat((review_prompt + "\n\n【待审草稿】:\n" + draft_res)) 
    logging.info("阶段2完成，已生成修正版文本。")
    
    # === 阶段3：格式化与文件生成 ===
    logging.info("--- 阶段3：格式化与文件写入 ---")
    with open(os.path.join(cur_path, 'formatting.txt'), 'r', encoding='utf-8') as f:
        formatting_prompt = f.read()
    
    agent3_name = "formatting" 
    formatting_agent = Ai.LLM(conf[agent3_name]["api_key"], conf[agent3_name]["base_url"], conf[agent3_name]["model"], "你是一个内容解析与文件系统格式化引擎")
    json_res = formatting_agent.chat((formatting_prompt + "\n\n【待解析文本】:\n" + corrected_res)) 
    logging.info("阶段3完成，已生成JSON。")

    # === 最后一步：写入文件系统 ===
    logging.info("--- 开始写入文件 ---")
    output_directory = os.path.join(cur_path, 'Obsidian-Notes') 
    save_files_from_json(json_res, output_directory)
    
    # === 清理工作 ===
    logging.info("--- 清理已处理的图片 ---")
    del_images(image_paths)
    
    logging.info("--- 所有任务完成 ---")
    
def v025():
    # 配置文件读取
    cur_path = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(cur_path, 'keyexample.json')
    with open(example_path,"r") as f:
        default_cfg = json.load(f)
    ai_cfg = file.Config("LLM_settings",default_cfg) 
    conf = ai_cfg.context
    
    # Vision 与 Review 生成与改进
    
    # init
    vision_prompt = prompt_reader("vision.txt")
    review_prompt = prompt_reader("review-cyc.txt")
    
    vision_agent_name = "vision"
    vision_agent = Ai.LLM(conf[vision_agent_name]["api_key"], conf[vision_agent_name]["base_url"], conf[vision_agent_name]["model"], "你是一名高中生物老师")
    review_agent_name = "review"
    review_agent = Ai.LLM(conf[review_agent_name]["api_key"], conf[review_agent_name]["base_url"], conf[review_agent_name]["model"], "你是一名高中生物老师")

    #=== 阶段1：视觉识别与草稿生成 ===
    
    # 图片编码
    logging.info("--- 阶段1：视觉识别与草稿生成 ---")
    image_paths = get_image_paths()
    if not image_paths:
        logging.warning("未找到任何图片，程序退出。")
        exit()
    images_encoded = []
    for image_path in image_paths: 
        images_encoded.append(Ai.encode_image(image_path))
        
    # 草稿生成
    draft_res = vision_agent.chat(vision_prompt, images_encoded)
    logging.info("阶段1完成，已生成草稿。")
    

    # === 阶段2：校对与修正 ===
    logging.info("--- 阶段2：校对与修正 ---")
    review_res = review_agent.chat((review_prompt + "\n\n【待审草稿】:\n" + draft_res))
    logging.info("阶段2完成，已生成修正版文本。")
    
    end_res = vision_agent.chat(review_res)

def v030():
    # --- 1. 配置与初始化 (主线程) ---
    logging.info("--- 初始化配置与AI Agent ---")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    conf = load_config(os.path.join(cur_path, 'key-api.json'))
    
    # 按照你的文件名读取Prompts
    prompt_stage1 = prompt_reader("vision.txt")
    prompt_stage2_template = prompt_reader("build.txt")
    prompt_stage3_template = prompt_reader("gen.txt")
    
    vision_agent = Ai.LLM(conf["vision"]["api_key"], conf["vision"]["base_url"], conf["vision"]["model"], "你是一个生物学概念与上下文提取器")
    build_agent = Ai.LLM(conf["review"]["api_key"], conf["review"]["base_url"], conf["review"]["model"], "你是一个知识图谱架构师")
    gen_agent = Ai.LLM(conf["formatting"]["api_key"], conf["formatting"]["base_url"], conf["formatting"]["model"], "你是一个原子化笔记撰写引擎")

    output_directory = os.path.join(cur_path, 'Obsidian-Notes')
    os.makedirs(output_directory, exist_ok=True)

    # --- 2. 扇出 (Fan-out): 并行执行视觉提取 ---
    image_paths = get_image_paths()
    if not image_paths:
        logging.warning("未找到任何图片，程序退出。")
        return
        
    logging.info(f"--- 阶段1 (并行): 发现 {len(image_paths)} 张图片，启动最多5个线程进行视觉提取 ---")
    
    all_markdown_outputs = []
    # 使用ThreadPoolExecutor来管理线程池和收集结果
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_image = {executor.submit(run_vision_stage_for_image, img_path, vision_agent, prompt_stage1): img_path for img_path in image_paths}
        
        for future in concurrent.futures.as_completed(future_to_image):
            try:
                result_markdown = future.result()
                if result_markdown and result_markdown.strip():
                    all_markdown_outputs.append(result_markdown)
            except Exception as exc:
                image_name = os.path.basename(future_to_image[future])
                logging.error(f"处理图片 {image_name} 的结果时产生异常: {exc}")

    logging.info("--- 所有视觉提取线程已完成 ---")

    # --- 3. 扇入 (Fan-in): 聚合所有Markdown结果 ---
    if not all_markdown_outputs:
        logging.warning("所有图片均未能提取到有效内容，程序终止。")
        del_images(image_paths) 
        return


    aggregated_markdown_text = "\n\n---\n\n".join(all_markdown_outputs)
    logging.info(f"--- 阶段2 (聚合): 已将所有图片的上下文内容合并为一个文档 ---")
    # logging.debug(f"聚合后的Markdown内容:\n{aggregated_markdown_text}")

    # --- 4. 顺序执行: 知识建构 (Build/Architect) ---
    logging.info("--- 阶段3 (顺序): 开始基于聚合的Markdown内容进行知识建构 ---")
    try:
        architect_input_prompt = f"{prompt_stage2_template}\n\n# 输入的Markdown上下文\n{aggregated_markdown_text}"
        file_blueprints_json = build_agent.chat(architect_input_prompt)
        file_blueprints = json.loads(file_blueprints_json)
        logging.info(f"成功生成 {len(file_blueprints)} 个文件的全局蓝图。")
    except Exception as e:
        logging.error(f"阶段3: 解析文件蓝图失败: {e}\n原始输出:\n{file_blueprints_json}")
        del_images(image_paths) 
        return

    # --- 5. 顺序执行: 内容生成 (Gen/Writer) ---
    logging.info("--- 阶段4 (顺序): 开始根据全局蓝图生成文件内容 ---")
    final_files_to_write = []
    for i, blueprint in enumerate(file_blueprints):
        filename = blueprint.get("filename", f"untitled_{i}.md")
        logging.info(f"正在生成文件 ({i+1}/{len(file_blueprints)}): {filename}")
        try:
            writer_input_prompt = f"{prompt_stage3_template}\n\n# 文件蓝图 (JSON):\n{json.dumps(blueprint, ensure_ascii=False)}"
            markdown_content = gen_agent.chat(writer_input_prompt)
            markdown_content = re.sub(r'^```markdown\s*|\s*```$', '', markdown_content, flags=re.MULTILINE).strip()
            final_files_to_write.append({"filename": filename, "content": markdown_content})
        except Exception as e:
            logging.error(f"为 {filename} 生成内容时出错: {e}")
            continue

    # --- 6. 文件写入与清理 ---
    if final_files_to_write:
        save_files_from_json(json.dumps(final_files_to_write, ensure_ascii=False, indent=2), output_directory)
    
    logging.info("--- 清理已处理的图片 ---")
    del_images(image_paths)
    
    logging.info("--- 所有任务完成 ---")

    





if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    
    v030()