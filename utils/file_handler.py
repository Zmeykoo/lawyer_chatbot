from datetime import date
import os
from pathlib import Path

from PyPDF2 import PdfReader
import docx


class Reader:
    @staticmethod
    def read_pdf(file_path):
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text

    @staticmethod
    def read_word(file_path):
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    @staticmethod
    def read_txt(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text

    @staticmethod
    def read_documents_from_directory(directory):
        combined_text = ""
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".pdf"):
                combined_text += Reader.read_pdf(file_path)
            elif filename.endswith(".docx"):
                combined_text += Reader.read_word(file_path)
            elif filename.endswith(".txt"):
                combined_text += Reader.read_txt(file_path)
        return combined_text

class Writer:
    @staticmethod
    def write_txt(text, query_name):
        save_dir = Path("response")
        save_dir.mkdir(exist_ok=True)
        date_str = date.today().isoformat()

        num = 0
        while (path := save_dir / f"{query_name}_{date_str}_({num}).txt").exists():
            num += 1

        path.write_text(text, encoding="utf-8")