import os
from dotenv import load_dotenv
from typing import Annotated
from pydantic_settings import BaseSettings 
from pydantic import Field 

load_dotenv(".env")

class Config(BaseSettings):
    chunking_method : Annotated[str , Field()] = os.getenv("CHUNKING_METHOD") or "fixed"
    emedding_method : Annotated[str , Field()] = os.getenv("EMBEDDING_METHOD") or "static"
    vector_store    : Annotated[str , Field()] = os.getenv("VECTORE_STORE") or "pgvector"
    system_prompt   : Annotated[str , Field()] = os.getenv("SYSTEM_PROMPT") or "ANSWER THE QUERY WITH THE GIVEN CONTEXT"


