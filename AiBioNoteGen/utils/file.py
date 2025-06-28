"""文件操作工具模块

该模块提供Config类用于管理JSON格式的配置文件，支持自动创建、加载和更新配置文件。
主要功能包括：
- 自动创建不存在的配置文件
- 支持加密和非加密配置文件
- 提供配置加载和更新方法
"""

import os
import json
import logging

current_dir = os.path.dirname(__file__)
# log
def log_init():
    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(current_dir, '..', 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'file.log')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )


class Config:
    """JSON配置文件管理类
    
    用于创建、加载和更新JSON格式的配置文件。
    
    Attributes:
        path (str): 配置文件路径
        context (dict): 配置内容字典
    """
    def __init__ (self, name:str, default_context:dict = {}, secret:bool = True):
        """初始化Config实例
        
        Args:
            name (str): 配置文件名(不带扩展名)
            default_context (dict): 默认配置内容，默认为空字典
            secret (bool): 是否为加密配置文件，默认为True
        """
        if secret:
            self.path = os.path.join(current_dir, '..', f'secret-{name}.json')
        else:
            self.path = os.path.join(current_dir, '..', f'{name}.json')
        self.context = default_context
        try:
            if not os.path.exists(self.path):
                with open(self.path, 'w') as file:
                    file.write(json.dumps(self.context))  
        except Exception as e:
            logging.error(f"文件写入失败: {e}")
            raise e
        self.load()
            
    def load(self):
        """从文件加载配置内容
        
        读取JSON文件内容并更新到context属性
        
        Raises:
            JSONDecodeError: 当JSON文件格式错误时抛出
            IOError: 当文件读取失败时抛出
        """
        with open(self.path, 'r') as file:
            self.context = json.load(file)
            
    def update(self):
        """更新配置文件
        
        将当前context内容写入配置文件
        
        Raises:
            IOError: 当文件写入失败时抛出
        """
        self.context.update(self.context)
        with open(self.path, 'w') as file:
            file.write(json.dumps(self.context))
            

if __name__ == '__main__':
    log_init()
    config = Config('test')
    config.context['test'] = 'test'
    config.update()
    config.load()
    print(config.context)
