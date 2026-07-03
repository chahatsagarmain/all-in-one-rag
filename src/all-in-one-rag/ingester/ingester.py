from utils.utils import read_file , read_markdown
from faster_whisper import WhisperModel
from pypdf import PdfReader
from models.ingestor import Ingester , Content


class DataSourceIngester(Ingester):

    def _check_data_type(self , path : str) -> str:
        file_ext = path.split(".")[-1]
        if file_ext in ["pdf" , "md" , "txt"]:
            return file_ext
        return ""

    def ingest_data_from_path(self , path : str) -> Content:
        data_type = self._check_data_type(path)
        if data_type == "":
            raise TypeError("File Type is Unsupported")

        elif data_type == "pdf":
            pdf = PdfReader(stream=path)
            content = ""
            for page in pdf.pages:
                content += page.extract_text()
            metadata = {}
            if pdf.metadata is not None:
                for key , val in pdf.metadata.items():
                    metadata[key] = val
            return Content(text = content , metadata = metadata)

        elif data_type == "md":
            return read_markdown(path)
        else:
            return read_file(path)

class AudioIngester(Ingester):

    def __init__(self):
        self._model = WhisperModel(
            model_size_or_path="small",
            device="cpu",
            compute_type="int8",
            cpu_threads=88
        )

    def _check_data_type(self, path):
        file_ext = path.split(".")[-1]
        if file_ext in ["mp3" , "wav" , "webm"]:
            return file_ext
        return ""

    def ingest_data_from_path(self, path):
        data_type = self._check_data_type(path)
        if data_type == "":
            raise TypeError("Invalid audio file type")
        segments , info = self._model.transcribe(
            path
        )
        print(f"audio file info : {info}")

        content = "".join([seg.text for seg in segments]).strip()
        metadata = {"filename" : path.split("/")[-1]}
        return Content(text=content , metadata=metadata)        