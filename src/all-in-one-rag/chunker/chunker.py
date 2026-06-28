from models.chunker import Chunker
from chonkie import TokenChunker, SemanticChunker as ChonkieSemanticChunker
from models.ingestor import Content

class FixedSizeChunker(Chunker):
    
    def __init__(self, content: Content, chunk_size: int = 100, overlap: int = 10, tokenizer: str = "character"):
        super().__init__(content)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = tokenizer

    def create_chunks(self):
        chunker = TokenChunker(
            tokenizer=self._tokenizer,
            chunk_size=self._chunk_size,
            chunk_overlap=self._overlap
        )
        chunks = chunker(self._content.text)
        return {'chunks' :  [c[0].text if isinstance(c , list) else c.text for c in chunks], 'metadata' : self._content.metadata}
    

class SemanticChunker(Chunker):
    def __init__(self, content: Content, embedding_model: str = "minishlab/potion-base-32M", threshold: float = 0.8, chunk_size: int = 2048):
        super().__init__(content)
        self._embedding_model = embedding_model
        self._threshold = threshold
        self._chunk_size = chunk_size

    def create_chunks(self):
        chunker = ChonkieSemanticChunker(
            embedding_model=self._embedding_model,
            threshold=self._threshold,
            chunk_size=self._chunk_size
        )
        chunks = chunker(self._content.text)
        return {'chunks' : [c[0].text if isinstance(c , list) else c.text for c in chunks] , 'metadata' : self._content.metadata}