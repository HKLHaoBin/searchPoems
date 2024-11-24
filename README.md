# 语义搜索古诗词
## 1. 项目概述
本项目使用向量数据库[Milvus](https://zilliz.com.cn/)的语义搜索功能，实现了通过白话文搜索语义相似的古诗词的功能。
古诗词数据集在[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)基础上做了改动。

## 2. 安装
### 2.1 安装 Docker
Milvus 运行在 Docker 容器中，因此需要先安装 Docker Desktop。

Milvus 运行在 docker 容器中，所以需要先安装 Docker Desktop。
MacOS 系统安装方法：[Install Docker Desktop on Mac](https://docs.docker.com/desktop/install/mac-install/)。
Windows 系统安装方法：[Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)。

### 2.2 安装向量数据库 Milvus
下载并运行 Milvus 独立版脚本：

下载安装脚本：
```shell
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```

运行 Milvus：
```shell
bash standalone_embed.sh start
```

### 2.3 创建 python 虚拟环境（可选）
建议创建虚拟环境以避免依赖冲突：

```shell
python3 -m venv myenv
```

激活虚拟环境
```shell
source myenv/bin/activate
```

退出虚拟环境：
```shell
deactivate
```

### 2.4 安装 Python 依赖
```shell
pip install -r requirements.txt
```

## 3. 运行项目
执行 main.py 交互程序。
```shell
python main.py
```

支持以下四种命令：
`create`：在向量数据库中创建集合，生成指定文本的向量，导入集合中，并且创建索引。文件存放在`data`文件夹中，为唐诗数据集。`data_split_total`文件夹中出了唐诗数据集外，还包括宋词（SongCi.json）、宋诗（SongShi.json）和元曲（YuanQu.json）等数据集。如果希望搜索更多古诗词，可以把它们复制到`data`文件夹中，用来创建集合。但是创建集合的时间会更长。

`search`：进入搜索模式。在搜索模式中，输入句子搜索语义相近的古诗词，比如，`今天天气不错`。或者输入“句子 作者”，指定古诗词的作者，比如`今天天气不错 李白`。

```
搜索你想用古诗词表达的意思：中午吃点啥
title： 慵不能
author： 白居易
paragraphs：午后恣情寝，午时随事餐。
distance: 0.619
--------------------------------------------------
title： 与鲜于庶子自梓州成都少尹自襃城同行至利州道中作
author：岑参
paragraphs：过午方始饭，经时旋及瓜。
distance: 0.582
--------------------------------------------------
title： 夏日闲放
author： 白居易
paragraphs：午餐何所有，鱼肉一两味。
distance: 0.574
--------------------------------------------------
title： 苔崔侍郎钱舍人书问因继以诗author： 白居易
paragraphs：旦暮两蔬食，日中一闲眠。
distance: 0.559
--------------------------------------------------
```

其中`distance`的值表示该古诗词与输入句子的相似程度，值越接近1相似程度越高。

`delete`：删除已有集合。

`exit`：退出当前命令，或者退出项目。


## 4. 许可证
本项目采用 MIT 许可证。