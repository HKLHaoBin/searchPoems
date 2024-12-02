import os
import yaml
from github import Github
from searcher import Searcher

# 从 GitHub Token 获取 repo 信息
g = Github(os.getenv("GITHUB_TOKEN"))

def read_yaml_config(file_path):
    """读取YAML配置文件并返回配置"""
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return config_data

def format_search_results(results):
    """格式化搜索结果为字符串"""
    formatted_results = ""
    for result in results:
        formatted_results += f"title: {result['title']}\n"
        formatted_results += f"author: {result['author']}\n"
        formatted_results += f"paragraphs: {result['paragraphs']}\n"
        formatted_results += f"distance: {result['distance']}\n"
        formatted_results += "-"*50 + "\n"
    return formatted_results

def handle_issue_comment(issue, comment_body):
    """处理 Issue 或评论中的搜索命令"""
    # 从配置文件中加载 Milvus 配置信息
    config = read_yaml_config('config.yaml')
    searcher = Searcher(config)
    
    # 执行搜索
    results = searcher.search(comment_body)
    
    # 格式化并返回结果
    if results:
        result_text = format_search_results(results)
    else:
        result_text = "未找到相似的古诗词，请尝试其他词语。"
    
    # 在 Issue 中创建评论
    issue.create_comment(result_text)

def handle_event(event):
    """根据事件类型处理 GitHub Issue 输入"""
    issue = event['issue']  # 获取 Issue 对象

    if event['action'] in ["opened", "edited"]:
        # 如果是 Issue 打开或编辑，处理 Issue 输入
        handle_issue_comment(issue, issue.body)
    elif event['comment'] and event['action'] == "created":
        comment_body = event['comment']['body'].strip()
        if comment_body.lower().startswith('search'):
            # 如果评论中包含搜索命令，执行搜索
            search_query = comment_body.split(' ', 1)[1] if len(comment_body.split(' ', 1)) > 1 else ''
            handle_issue_comment(issue, search_query)

if __name__ == '__main__':
    # 获取 GitHub 事件数据
    event_name = os.getenv("GITHUB_EVENT_NAME")
    event_path = os.getenv("GITHUB_EVENT_PATH")

    with open(event_path, 'r') as f:
        event_data = yaml.safe_load(f)

    handle_event(event_data)
