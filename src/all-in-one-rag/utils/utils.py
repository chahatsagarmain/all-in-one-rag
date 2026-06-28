from pathlib import Path
from bs4 import BeautifulSoup
from markdown import markdown
from typing import List
from models.ingestor import Content

def read_file(path) -> Content:
    content = ""
    for line in read_file_generator(path):
        content += line
    file_path = Path(path)
    metadata = {
        "File Name": file_path.name,
        "Extension": file_path.suffix,
    }
    return Content(text = content , metadata = metadata)

def read_file_generator(path : str):
    with open(path , "r" , encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def read_markdown(path : str) -> Content:
    content = ""
    for line in read_file_generator(path):
        html = markdown(line)
        soup = BeautifulSoup(html, features="html.parser")
        content += soup.get_text()
    file_path = Path(path)
    metadata = {
        "File Name": file_path.name,
        "Extension": file_path.suffix,
    }
    return Content(text = content , metadata = metadata)
