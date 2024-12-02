import torch
import time
import json
from typing import List
import os
from tqdm import tqdm
from pymilvus import DataType, MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import requests  # 用于调用 Deepseek API

class Searcher:
    def __init__(self, config, api_key=None):
        self.embed_model = config.embedding.embed_model
        self.dim = config.embedding.dim
        self.uri = f"http://{config.milvus.host}:{config.milvus.port}"
        self.collection_name = config.milvus.collection_name
        self.limit = config.milvus.limit
        self.api_key = api_key  # Deepseek API 密钥

        # 创建 Milvus 客户端
        self.milvus_client = MilvusClient(uri=self.uri)

    def create_collection(self, collection_name):
        """创建集合"""
        # 检查同名集合是否存在，如果存在则删除，不存在则创建
        if self.milvus_client.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已经存在")
            try:
                self.milvus_client.drop_collection(self.collection_name)
                print(f"删除集合：{self.collection_name}")
            except Exception as e:
                print(f"删除集合时出现错误: {e}")
        # 创建集合模式
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            num_partitions=16,
            description=""
        )
        # 添加字段到 schema
        schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="paragraphs", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, max_length=32)
        schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="dense_vectors", datatype=DataType.FLOAT_VECTOR, dim=512)
        # 创建集合
        try:
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                shards_num=2
            )
            print(f"创建集合：{self.collection_name}")
        except Exception as e:
            print(f"创建集合的过程中出现了错误: {e}")
        # 等待集合创建成功
        while not self.milvus_client.has_collection(self.collection_name):
            time.sleep(1)
        print(f"集合 {self.collection_name} 创建成功")

    def vectorize_query(self, query):
        """向量化文本列表"""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_fp16 = device.startswith("cuda")
        bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=self.embed_model,
            device=device,
            use_fp16=use_fp16
        )
        vectors = bge_m3_ef.encode_documents(query)
        return vectors

    def get_files_from_dir(self, input_dir_path):
        """从目录中获取所有 JSON 文件"""
        file_paths = []
        for root, dirs, files in os.walk(input_dir_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths

    def vectorize_and_import_data(self, input_file_path, field_name, embed_model, batch_size):
        """向量化并导入数据"""
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)
            for data in data_list:
                data[field_name] = data[field_name][0]
            query = [data[field_name] for data in data_list]

        # 向量化文本数据
        vectors = self.vectorize_query(query)
        for data, dense_vectors in zip(data_list, vectors['dense']):
            data['dense_vectors'] = dense_vectors.tolist()
        print(f"正在将数据插入集合：{self.collection_name}")
        total_count = len(data_list)
        with tqdm(total=total_count, desc="插入数据") as pbar:
            for i in range(0, total_count, batch_size):
                batch_data = data_list[i:i + batch_size]
                res = self.milvus_client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
                pbar.update(len(batch_data))

    def create_index(self, collection_name):
        """创建索引"""
        print("正在创建索引")
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            index_name="IVF_FLAT",
            field_name="dense_vectors",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        self.milvus_client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        print("索引创建完成")
        print(f"正在加载集合：{self.collection_name}")
        self.milvus_client.load_collection(collection_name=self.collection_name)
        state = str(self.milvus_client.get_load_state(collection_name=self.collection_name)['state'])
        if state == 'Loaded':
            print("集合加载完成")
        else:
            print("集合加载失败")

    def create_vector_db(self):
        """创建向量数据库"""
        self.create_collection(self.collection_name)
        start_time = time.time()
        batch_size = 1000
        field_name = "paragraphs"
        input_dir_path = 'data'
        input_file_paths = self.get_files_from_dir(input_dir_path)
        for input_file_path in input_file_paths:
            print(f"正在处理文件：{input_file_path}")
            self.vectorize_and_import_data(input_file_path, field_name, self.embed_model, batch_size)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"数据入库耗时：{elapsed_time:.2f} 秒")
        self.create_index(self.collection_name)

    def search(self, query):
        """使用 Deepseek API 和 Milvus 进行搜索"""
        if self.api_key:
            # 如果提供了 API 密钥，先调用 Deepseek API 进行搜索
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.post(
                "https://api.deepseek.com/v1/search",
                headers=headers,
                json={"query": query}
            )
            if response.status_code == 200:
                print(f"Deepseek API 返回：{response.json()}")
                return response.json()['results']
            else:
                print(f"Deepseek API 错误：{response.status_code}")
        
        # 如果没有 API 密钥，继续使用 Milvus 进行传统向量搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 16,
                "radius": 0.1,
                "range_filter": 1
            }
        }
        query_vectors = [self.vectorize_query([query])['dense'][0].tolist()]
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_vectors,
            anns_field="dense_vectors",
            search_params=search_params,
            limit=self.limit,
            output_fields=["paragraphs", "title", "author"]
        )
        self.print_vector_results(res)

    def search_filter_by_author(self, query, author):
        """根据作者进行过滤搜索"""
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 16,
                "radius": 0.1,
                "range_filter": 1
            }
        }
        query_vectors = [self.vectorize_query([query])['dense'][0].tolist()]
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_vectors,
            anns_field="dense_vectors",
            search_params=search_params,
            limit=self.limit,
            output_fields=["paragraphs", "title", "author"],
            filter=f"author =='{author}'"
        )
        self.print_vector_results(res)

    def print_vector_results(self, res):
        """打印搜索结果并格式化为可返回的字符串"""
        results = []
        for hits in res:
            for hit in hits:
                entity = hit.get("entity")
                results.append({
                    "title": entity['title'],
                    "author": entity['author'],
                    "paragraphs": entity['paragraphs'],
                    "distance": hit['distance']
                })
        return results

    def delete_collection(self):
        """删除集合"""
        if self.milvus_client.has_collection(self.collection_name):
            try:
                self.milvus_client.drop_collection(self.collection_name)
                print(f"删除集合：{self.collection_name}")
            except Exception as e:
                print(f"删除集合时出现错误: {e}")
        else:
            print(f"集合 {self.collection_name} 不存在，无需删除。")
