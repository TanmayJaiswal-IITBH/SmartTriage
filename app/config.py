# isko settings me dal dena ache se

from pydantic import BaseModel

class settings(BaseModel):

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHROMA_PATH =  "./data/chroma"
    COLLECTION_NAME = "openlake_issues"

Setting = settings() # iska name mat change karna