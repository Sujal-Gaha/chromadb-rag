from fastapi import File, UploadFile
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class UploadFilesRequest(BaseModel):
    files: list[UploadFile] = File(...)
