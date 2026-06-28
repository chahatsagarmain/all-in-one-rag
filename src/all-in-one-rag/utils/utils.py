from bs4 import BeautifulSoup
from markdown import markdown

def read_file(path) -> str:
    content = ""
    for line in read_file_generator(path):
        content += line
    return content

def read_file_generator(path : str):
    with open(path , "r" , encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def read_markdown(path : str) -> str:
    content = ""
    for line in read_file_generator(path):
        html = markdown(line)
        soup = BeautifulSoup(html, features="html.parser")
        content += soup.get_text()
    return content