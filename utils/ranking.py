# !pip install ragatouille -q

from sentence_transformers import SentenceTransformer
from voyager import Index, Space

import joblib
from typing import List, Dict
from rank_bm25 import BM25Okapi
import pandas as pd


class MyExistingRetrievalPipeline:
    index: Index
    embedder: SentenceTransformer
    bm25: BM25Okapi
    df: pd.DataFrame  # Add a new attribute to store the DataFrame

    def __init__(self, embedder_name: str = "intfloat/multilingual-e5-large"):
        self.embedder = SentenceTransformer(embedder_name)
        self.collection_map_44 = {}
        self.collection_map_223 = {}
        self.collection_map_others = {}
        self.index_44 = Index(
            Space.Cosine,
            num_dimensions=self.embedder.get_sentence_embedding_dimension(),
        )
        self.index_223 = Index(
            Space.Cosine,
            num_dimensions=self.embedder.get_sentence_embedding_dimension(),
        )
        self.index_others = Index(
            Space.Cosine,
            num_dimensions=self.embedder.get_sentence_embedding_dimension(),
        )
        self.doc_texts = []

    def index_documents(self, df_44: pd.DataFrame, df_223: pd.DataFrame, df_others: pd.DataFrame, batch_size: int = 32) -> None:
        self.df_223 = df_223.reset_index(drop=True)
        self.df_44 = df_44.reset_index(drop=True)
        self.df_others = df_others.reset_index(drop=True)
        
        self.doc_texts_44 = self.df_44['passage_full'].tolist()
        self.doc_texts_223 = self.df_223['passage_full'].tolist()
        self.doc_texts_others = self.df_others['passage_full'].tolist()
        
        for i in range(0, len(self.df_44), batch_size):
            batch = self.df_44.iloc[i:i+batch_size]
            documents = batch['passage_full'].tolist()
            embeddings = self.embedder.encode(documents, show_progress_bar=True)
            for j, embedding in enumerate(embeddings):
                doc_id = batch.iloc[j]['passage']
                self.collection_map_44[self.index_44.add_item(embedding)] = {
                    'doc_id': doc_id,
                    'content': documents[j]
                }
        
        for i in range(0, len(self.df_223), batch_size):
            batch = self.df_223.iloc[i:i+batch_size]
            documents = batch['passage_full'].tolist()
            embeddings = self.embedder.encode(documents, show_progress_bar=True)
            for j, embedding in enumerate(embeddings):
                doc_id = batch.iloc[j]['passage']
                self.collection_map_223[self.index_223.add_item(embedding)] = {
                    'doc_id': doc_id,
                    'content': documents[j]
                }

        for i in range(0, len(self.df_others), batch_size):
            batch = self.df_others.iloc[i:i+batch_size]
            documents = batch['passage_full'].tolist()
            embeddings = self.embedder.encode(documents, show_progress_bar=True)
            for j, embedding in enumerate(embeddings):
                doc_id = batch.iloc[j]['passage']
                self.collection_map_others[self.index_others.add_item(embedding)] = {
                    'doc_id': doc_id,
                    'content': documents[j]
                }
                
    def save_index(self, index_file_path_44: str, map_file_path_44: str, index_file_path_223: str, map_file_path_223: str, index_file_path_others: str, map_file_path_others: str):
        """Saves the index to a file and the collection map using joblib."""
        self.index_44.save(index_file_path_44) 
        joblib.dump(self.collection_map_44, map_file_path_44)
        self.index_223.save(index_file_path_223) 
        joblib.dump(self.collection_map_223, map_file_path_223) 
        self.index_others.save(index_file_path_others) 
        joblib.dump(self.collection_map_others, map_file_path_others)

        


    def load_index(self, index_file_path_44: str, map_file_path_44: str, index_file_path_223: str, map_file_path_223: str, index_file_path_others: str, map_file_path_others: str):
        """Loads the index from a file and the collection map using joblib."""
        index_44 = Index.load(index_file_path_44)
        self.index_44 = index_44  
        self.collection_map_44 = joblib.load(map_file_path_44)
        
        index_223 = Index.load(index_file_path_223)
        self.index_223 = index_223  
        self.collection_map_223 = joblib.load(map_file_path_223) 
        
        index_others = Index.load(index_file_path_others)
        self.index_others = index_others  
        self.collection_map_others = joblib.load(map_file_path_others) 
        


    def query(self, selected_law, query: str, k: int = 10) -> List[Dict[str, str]]:
        query_embedding = self.embedder.encode(query)

        dense_results_44 = self.index_44.query(query_embedding, k=k)[0]
        dense_docs_44 = [{'doc_id': self.collection_map_44[idx]['doc_id'], 'content': self.collection_map_44[idx]['content']} for idx in dense_results_44]
        
        dense_results_223 = self.index_223.query(query_embedding, k=k)[0]
        dense_docs_223 = [{'doc_id': self.collection_map_223[idx]['doc_id'], 'content': self.collection_map_223[idx]['content']} for idx in dense_results_223]
        
        dense_results_others = self.index_others.query(query_embedding, k=k)[0]
        dense_docs_others = [{'doc_id': self.collection_map_others[idx]['doc_id'], 'content': self.collection_map_others[idx]['content']} for idx in dense_results_others]
        
        if selected_law == '223-ФЗ':
            combined_docs = dense_docs_223 + dense_docs_others
            new_k = 2 * k
        elif selected_law == '44-ФЗ':
            combined_docs = dense_docs_44 + dense_docs_others
            new_k = 2 * k
        else:
            combined_docs = dense_docs_44 + dense_docs_223 + dense_docs_others
            new_k = 3 * k
            
        combined_docs = list(set([tuple(doc.items()) for doc in combined_docs]))
        combined_docs = [dict(doc) for doc in combined_docs]
#         new_k = 3 * k

        
        return combined_docs[:new_k]

