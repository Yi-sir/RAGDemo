import torch
from elasticsearch import Elasticsearch
from transformers import BertModel, BertTokenizer


class Retriever:
    def __init__(self):
        self.es = Elasticsearch()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.index_name = "documents"  # 假设你的Elasticsearch索引名称为'documents'

    def _encode_query(self, query):
        inputs = self.tokenizer(
            [query], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def retrieve(self, question):
        query_vector = self._encode_query(question)
        search_query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector.tolist()},
                    },
                }
            },
        }
        response = self.es.search(index=self.index_name, body=search_query)
        contexts = [hit["_source"]["text"] for hit in response["hits"]["hits"]]
        return contexts
