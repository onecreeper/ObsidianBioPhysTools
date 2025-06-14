import base64
import json
import requests
import logging
import os
from openai import OpenAI
from typing import List, Optional, Union

# log
current_dir = os.path.dirname(__file__)
log_dir = os.path.join(current_dir, '..', 'log')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'Ai.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)


class LLM:
    def __init__(self, api_key, base_url, model_name, system_prompt = "使用中文回答用户的问题"):
        
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.messages = [{"role": "system", "content": system_prompt}]

    def format_chinese_response(self, text):
        """格式化中文回答"""
        # 分段处理
        paragraphs = text.split('\n')
        formatted_paragraphs = []
        
        for p in paragraphs:
            if p.strip():
                # 为每个段落添加缩进和格式
                formatted_p = "    " + p.strip()
                formatted_paragraphs.append(formatted_p)
        
        # 添加装饰性边框
        width = max(len(p) for p in formatted_paragraphs) if formatted_paragraphs else 20
        border = "━" * width
        
        formatted_text = f"\n{border}\n"
        for p in formatted_paragraphs:
            formatted_text += f"{p}\n"
        formatted_text += f"{border}\n"
        
        return formatted_text
    
    def _prepare_message_content(self, text: str, images: Optional[Union[str, List[str]]] = None):
        """准备消息内容，支持纯文本或文本+图片
        
        Args:
            text: 文本内容
            images: 单个图片或图片列表，可以是URL或base64编码
        
        Returns:
            格式化后的消息内容
        """
        if images is None:
            return text

        if isinstance(images, str):
            images = [images]
        
        content = [{"type": "text", "text": text}]
        
        for img in images:
            if img.startswith("http://") or img.startswith("https://"):
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": img}
                })
            elif img.startswith("data:image"):
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": img}
                })
            else:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
        return content
        
    def chat(self, text: str, images: Optional[Union[str, List[str]]] = None, streaming_output: bool = True) -> str | None:
        """与LLM进行对话，支持文本和图片输入

        Args:
            text (str): 文本消息
            images (Optional[Union[str, List[str]]]): 单个图片或图片列表，可以是URL或base64编码
            streaming_output (bool, optional): 是否流式输出. Defaults to True.

        Returns:
            str | None: ai的回答，None为异常
        """
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        # 准备消息内容
        message_content = self._prepare_message_content(text, images)
        self.messages.append({"role": "user", "content": message_content})

        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                stream=True,
            )
            
            if streaming_output:
                print("\n" + "="*50)
                print("🤔 思考过程:")
                print("="*50)
            
            for chunk in completion:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                # 处理思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                    if streaming_output:
                        print(delta.reasoning_content, end='', flush=True)
                else:
                    # 处理回复内容
                    if hasattr(delta, 'content') and delta.content is not None:
                        if delta.content != "" and is_answering is False:
                            is_answering = True
                            if streaming_output:
                                print("\n" + "="*50)
                                print("💡 回答结果:")
                                print("="*50)
                        answer_content += delta.content
                        if streaming_output:
                            print(delta.content, end='', flush=True)
            
            if streaming_output:
                print("\n" + "="*50)

            # 将AI的回复添加到消息历史中
            self.messages.append({"role": "assistant", "content": answer_content})
            
            # 记录日志
            log_msg = f"对话 - 用户: {text}"
            if images:
                img_count = 1 if isinstance(images, str) else len(images)
                log_msg += f" (包含{img_count}张图片)"
            logging.info(log_msg)
            logging.info(f"对话 - AI: {answer_content[:100]}...")
            
            formatted_answer = self.format_chinese_response(answer_content)
            return formatted_answer
            
        except Exception as e:
            logging.error(f"聊天时发生错误: {str(e)}")
            return None
    
    def ask(self, text: str, images: Optional[Union[str, List[str]]] = None, streaming_output: bool = True) -> str | None:
        """单次提问方法，不记录对话历史
        
        Args:
            text (str): 文本消息
            images (Optional[Union[str, List[str]]]): 单个图片或图片列表，可以是URL或base64编码
            streaming_output (bool, optional): 是否流式输出. Defaults to True.
        
        Returns:
            str | None: ai的回答，None为异常
        """
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        reasoning_content = ""
        answer_content = ""
        is_answering = False
        
        # 创建临时消息列表
        temp_messages = [{"role": "system", "content": self.messages[0]["content"]}]
        
        # 准备消息内容
        message_content = self._prepare_message_content(text, images)
        temp_messages.append({"role": "user", "content": message_content})
        
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=temp_messages,
                stream=True,
            )
            
            if streaming_output:
                print("\n" + "="*50)
                print("🤔 思考过程:")
                print("="*50)
            
            for chunk in completion:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                # 处理思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                    if streaming_output:
                        print(delta.reasoning_content, end='', flush=True)
                else:
                    # 处理回复内容
                    if hasattr(delta, 'content') and delta.content is not None:
                        if delta.content != "" and is_answering is False:
                            is_answering = True
                            if streaming_output:
                                print("\n" + "="*50)
                                print("💡 回答结果:")
                                print("="*50)
                        answer_content += delta.content
                        if streaming_output:
                            print(delta.content, end='', flush=True)
            
            if streaming_output:
                print("\n" + "="*50)
            
            # 记录日志
            log_msg = f"单次提问: {text}"
            if images:
                img_count = 1 if isinstance(images, str) else len(images)
                log_msg += f" (包含{img_count}张图片)"
            logging.info(log_msg)
            logging.info(f"AI回答: {answer_content[:100]}...")
            
            formatted_answer = self.format_chinese_response(answer_content)
            return formatted_answer
            
        except Exception as e:
            logging.error(f"单次提问时发生错误: {str(e)}")
            return None

    def clear_history(self):
        """清除对话历史，只保留系统提示"""
        self.messages = [self.messages[0]]
        logging.info("对话历史已清除")

    def get_history(self):
        """获取当前对话历史"""
        return self.messages

    def get_history_count(self):
        """获取对话轮数（不包括系统提示）"""
        return (len(self.messages) - 1) // 2


def encode_image(image_path: str) -> str:
    """将本地图片编码为base64
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        base64编码的字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def main():
    qvq = LLM(
        "sk-sbsbsbsbsbsbsbsbsbsbsbsbsbsbsbsbsbsbsbsb", 
        "https://dashscope.aliyuncs.com/compatible-mode/v1", 
        "qvq-max", 
        "使用中文回答用户的问题"
    )
    images = [encode_image("AiBioNoteGen/utils/1.jpg"), encode_image("AiBioNoteGen/utils/2.jpg")]
    qvq.chat("你好,介绍下这些图片",images)



if __name__ == '__main__':
    main()
