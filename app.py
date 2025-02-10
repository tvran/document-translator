import os
import streamlit as st
import zipfile
from lxml import etree
from pdf2docx import Converter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentTranslator:
    def __init__(self):
        # Retrieve Hugging Face token from environment variable
        hf_token = os.getenv('HF_TOKEN')
        
        # Load Tilmash model with token authentication
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "issai/tilmash", 
                use_auth_token=hf_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "issai/tilmash", 
                use_auth_token=hf_token
            )
            self.tilmash = pipeline(
                "translation", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                src_lang="rus_Cyrl", 
                tgt_lang="kaz_Cyrl", 
                max_length=1000
            )
        except Exception as e:
            st.error(f"Ошибка при загрузке модели: {str(e)}")
            st.error("Проверьте правильность HF_TOKEN в .env файле")
            raise

    def pdf_to_docx(self, pdf_path, docx_path):
        """Convert PDF to DOCX"""
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None, continuous=True)
        cv.close()
        return docx_path

    def translate_text(self, text):
        """Translate individual text segment"""
        return self.tilmash(text)[0]['translation_text'] if text.strip() else text

    def translate_docx(self, docx_path, output_path):
        """Translate DOCX while preserving formatting"""
        with zipfile.ZipFile(docx_path, "r") as docx:
            xml_content = docx.read("word/document.xml")

        tree = etree.fromstring(xml_content)

        # Translate text within XML tags
        for elem in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
            original_text = elem.text
            if original_text and len(original_text.strip()) > 0:
                elem.text = self.translate_text(original_text)

        # Save new DOCX
        with zipfile.ZipFile(output_path, "w") as docx:
            for item in zipfile.ZipFile(docx_path, "r").infolist():
                if item.filename == "word/document.xml":
                    docx.writestr(item.filename, etree.tostring(tree, encoding="utf-8"))
                else:
                    docx.writestr(item.filename, zipfile.ZipFile(docx_path, "r").read(item.filename))

        return output_path

def main():
    st.title("Переводчик документов Tilmash")
    
    # Initialize translator
    translator = DocumentTranslator()
    
    # Language selection
    translation_direction = st.selectbox(
        "Выберите направление перевода",
        ["Русский → Казахский", "Казахский → Русский"]
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Загрузите документ (PDF или DOCX)", 
        type=['pdf', 'docx']
    )
    
    if uploaded_file is not None:
        # Temporary file handling
        input_path = f"/tmp/input{os.path.splitext(uploaded_file.name)[1]}"
        output_path = f"/tmp/translated{os.path.splitext(uploaded_file.name)[1]}"
        
        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Translate button
        if st.button("Перевести документ"):
            try:
                # Handle PDF conversion if needed
                if uploaded_file.type == 'application/pdf':
                    converted_docx_path = f"/tmp/converted.docx"
                    translator.pdf_to_docx(input_path, converted_docx_path)
                    input_path = converted_docx_path
                
                # Translate document
                translated_path = translator.translate_docx(input_path, output_path)
                
                # Provide download button
                with open(translated_path, "rb") as file:
                    st.download_button(
                        label="Скачать переведенный документ",
                        data=file.read(),
                        file_name=f"translated_{uploaded_file.name}",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                
                st.success("Документ успешно переведен!")
            
            except Exception as e:
                st.error(f"Ошибка при переводе: {str(e)}")

if __name__ == "__main__":
    main()