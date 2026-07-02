from sentence_transformers import cross_encoder
from typing import List

class CrossEncoder():
    
    def __init__(self):
        self._model = cross_encoder.CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    def get_tok_k_documents(self , query : str , documents : List[str] , top_k : int = 3) -> List[str]:
        processed_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                processed_docs.append(doc.get("text", str(doc)))
            else:
                processed_docs.append(str(doc))

        scores = self._model.rank(
            query , processed_docs , top_k=top_k , convert_to_tensor=True , return_documents=True
            )
        res = []
        for itr in scores:
            res.append(itr['text'])
        return res