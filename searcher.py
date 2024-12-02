import torch
import time
import json
from typing import List
import os
from tqdm import tqdm
from pymilvus import DataType, MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

class Searcher:
    def __init__(self, config):
        self.embed_model = config.embedding.embed_model
        self.dim = config.embedding.dim
        self.uri = f"http://{config.milvus.host}:{config.milvus.port}"
        self.collection_name = config.milvus.collection_name
        self.limit = config.milvus.limit
        # 创建Milvus实例
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
        # 添加字段到schema
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
        file_paths = []
        # 遍历目录中的所有文件和子目录
        for root, dirs, files in os.walk(input_dir_path):
            for file in files:
                # 检查文件扩展名是否为 .json
                if file.endswith('.json'):
                    # 构建文件的完整路径
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths


    def vectorize_and_import_data(
        self, 
        input_file_path, 
        field_name, 
        embed_model,
        batch_size
        ):
        """读取json文件中的数据，向量化后，分批入库"""
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)
            # data_list = data_list[:1000]
            # paragraphs字段的值是列表，需要变成字符串以符合Milvus的要求
            for data in data_list:
                data[field_name] = data[field_name][0]
            # 提取该json文件中的所有指定字段的值
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
        # 创建索引参数
        print("正在创建索引")
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            index_name="IVF_FLAT",
            field_name="dense_vectors",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        # 创建索引
        self.milvus_client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        print("索引创建完成")
        # 加载集合
        print(f"正在加载集合：{self.collection_name}")
        self.milvus_client.load_collection(collection_name=self.collection_name)
        state = str(self.milvus_client.get_load_state(collection_name=self.collection_name)['state'])
        # 验证加载状态
        if state == 'Loaded':
            print("集合加载完成")
        else:
            print("集合加载失败")

    def create_vector_db(self):
        """创建向量数据库"""
        # 创建集合
        self.create_collection(self.collection_name)
        # 入库
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
        # 创建索引
        self.create_index(self.collection_name)

    def search(self, query):
        """搜索"""
        # 创建搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 16,
                "radius": 0.1,
                "range_filter": 1
            }
        }
        # 搜索向量
        query_vectors = [self.vectorize_query([query])['dense'][0].tolist()]
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_vectors,
            anns_field="dense_vectors",
            search_params=search_params,
            limit=self.limit,
            output_fields=["paragraphs", "title", "author"]
        )
        # 打印搜索结果
        self.print_vector_results(res)

    def search_filter_by_author(self, query, author):
        """搜索并且通过作者过滤"""
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 16,
                "radius": 0.1,
                "range_filter": 1
            }
        }
        # 搜索向量
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
        # 打印搜索结果
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
        if self.milvus_client.has_collection(self.collection_name):
            try:
                self.milvus_client.drop_collection(self.collection_name)
                print(f"删除集合：{self.collection_name}")
            except Exception as e:
                print(f"删除集合时出现错误: {e}")
        else:
            print(f"集合 {self.collection_name} 不存在，无需删除。")
