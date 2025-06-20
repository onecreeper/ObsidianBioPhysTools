from utils import Ai
from utils import file
import base64
import json
import os
import requests
import logging
from openai import OpenAI
import re  

current_dir = os.path.dirname(__file__)
log_dir = os.path.join(current_dir, '..', 'log')
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



if __name__ == "__main__":
    v025()