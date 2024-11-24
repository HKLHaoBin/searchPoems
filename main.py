from searcher import Searcher
import yaml
# 导入easydict库用于将字典转换为更易用的对象
from easydict import EasyDict

def read_yaml_config(file_path):
    """读取YAML配置文件并返回EasyDict对象"""
    with open(file_path, "r") as file:
        # 读取yaml文件，得到字典
        config_data = yaml.safe_load(file)
    # 把字典转换为EasyDict对象。可以通过属性的形式访问键值对
    return EasyDict(config_data)

class CommandLine:
    def __init__(self, config_path):
        """初始化CommandLine类"""
        self._searcher = None
        self.config_path = config_path

    # 创建字符画的在线网站：http://patorjk.com/software/taag/
    def show_start_info(self):
        """显示启动信息"""
        with open('./start_info.txt') as fw:
            print(fw.read())

    def run(self):
        """运行命令行工具"""
        self.show_start_info()
        config = read_yaml_config(self.config_path)
        # 初始化searcher实例，用于搜索
        self._searcher = Searcher(config)
        self._show_instructions()
        self._handle_commands()

    def _show_instructions(self):
        """显示用户指令"""
        print('1. 输入`create`创建集合。')
        print('2. 创建完成后，输入`search`进入搜索模式，输入句子搜索古诗词，或者输入“句子 作者”指定古诗词的作者。')
        print('3. `delete`删除已有集合。')
        print('4. `exit`退出。')

    def _handle_commands(self):
        """处理用户输入的命令"""
        while True:
            user_input = input("(searcher) ")
            commands = user_input.split(' ')
            # 通过空格分成命令和参数两部分
            command = commands[0]
            args = commands[1:]
            # 进入create模式
            if command == 'create' and not args:
                self.create_vector_db()
            elif command == 'create':
                print('(searcher) create不接受参数。')
            # 进入search模式
            elif command == 'search' and not args:
                self.search()
            elif command == 'search':
                print('(searcher) search不接受参数。')
            # 删除集合
            elif command == 'delete' and not args:
                self.delete_collection()
            elif command == 'delete':
                print('(searcher) delete不接受参数。')
            # 退出应用
            elif command == 'exit':
                self._exit()
            else:
                print('(searcher) 只有[create|search|delete|exit]命令, 请重新尝试。')

    def create_vector_db(self):
        """创建集合"""
        self._searcher.create_vector_db()
        print('(searcher) 创建集合完成')

    def search(self):
        """搜索相似语句"""
        while True:
            user_input = input("(searcher) 搜索你想用古诗词表达的意思: ")
            commands = user_input.split(' ')
            # 通过空格分成命令和参数两部分
            command = commands[0]
            if len(commands) >=2:
                author = commands[1]
            if command == 'exit':
                print('(searcher) 退出问答')
                break
            # 如果命令为空，跳过本次循环
            elif not commands:
                print('(searcher) 请重新输入')
                continue
            elif len(commands) == 2:
                self._searcher.search_filter_by_author(command, author)
            elif len(commands) == 1:
                self._searcher.search(command)
            else:
                print('(searcher) search接受1到2个参数，第一个参数是想用古诗词表达的意思，第二个参数是指定古诗词的作者')

    def delete_collection(self):
        """删除集合"""
        self._searcher.delete_collection()

    def _exit(self):
        """退出程序"""
        exit()

if __name__ == '__main__':
    cli = CommandLine('config.yaml')
    cli.run()