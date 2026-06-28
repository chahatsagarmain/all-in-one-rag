from utils.utils import read_file , read_markdown
from markdown import markdown
from pypdf import PdfReader
from models.ingestor import Ingester


class DataSourceIngester(Ingester):

    def __init__(self , path : str):
        self.path = path

    def _check_data_type(self) -> str:
        path = self.path
        file_ext = path.split(".")[-1]
        if file_ext in ["pdf" , "md" , "txt"]:
            return file_ext
        return ""


    def ingest_data_from_path(self) -> str:
        path = self.path
        data_type = self._check_data_type()
        if data_type == "":
            raise TypeError("File Type is Unsupported")

        elif data_type == "pdf":
            pdf = PdfReader(stream=path)
            content = ""
            for page in pdf.pages:
                content += page.extract_text()
            return content

        elif data_type == "md":
            return read_markdown(path)
        else:
            return read_file(path)


    

        