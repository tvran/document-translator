import os
import streamlit as st
import zipfile
from lxml import etree
from pdf2docx import Converter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import tempfile
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_option('server.fileWatcherType', 'none')

# Load environment variables
load_dotenv()

# Create temp directory
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@st.cache_resource
def load_translation_model():
    """Load Tilmash translation model with Hugging Face token"""
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        st.error("Hugging Face токен не найден. Пожалуйста, установите HF_TOKEN в .env файле.")
        st.stop()
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "issai/tilmash", 
            token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "issai/tilmash", 
            token=hf_token
        )
        return pipeline("translation", model=model, tokenizer=tokenizer, 
                        src_lang="rus_Cyrl", tgt_lang="kaz_Cyrl", max_length=1000)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        st.stop()

def pdf_to_docx(pdf_path, docx_path):
    """Convert PDF to DOCX with error handling"""
    try:
        logger.info(f"Converting PDF: {pdf_path} to DOCX: {docx_path}")
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None, continuous=True)
        cv.close()
        logger.info("PDF conversion completed successfully")
        return docx_path
    except Exception as e:
        logger.error(f"Error converting PDF: {str(e)}")
        raise Exception(f"Ошибка при конвертации PDF: {str(e)}")

def translate_text(tilmash_pipeline, text):
    """Translate text using Tilmash model"""
    try:
        if not text.strip():
            return text
        return tilmash_pipeline(text)[0]['translation_text']
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        raise Exception(f"Ошибка при переводе текста: {str(e)}")

def translate_docx(tilmash_pipeline, docx_path, output_path, source_lang='rus', target_lang='kaz'):
    """Translate DOCX file while preserving formatting"""
    try:
        # Modify pipeline based on translation direction
        if source_lang == 'kaz':
            tilmash_pipeline.tokenizer.src_lang = "kaz_Cyrl"
            tilmash_pipeline.tokenizer.tgt_lang = "rus_Cyrl"
        else:
            tilmash_pipeline.tokenizer.src_lang = "rus_Cyrl"
            tilmash_pipeline.tokenizer.tgt_lang = "kaz_Cyrl"

        with zipfile.ZipFile(docx_path, "r") as docx:
            xml_content = docx.read("word/document.xml")
        
        tree = etree.fromstring(xml_content)
        
        # Translate only text within <w:t> tags
        for elem in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
            original_text = elem.text
            if original_text and len(original_text.strip()) > 0:
                elem.text = translate_text(tilmash_pipeline, original_text)
        
        # Save new DOCX
        with zipfile.ZipFile(output_path, "w") as docx:
            with zipfile.ZipFile(docx_path, "r") as original:
                for item in original.infolist():
                    if item.filename == "word/document.xml":
                        docx.writestr(item.filename, etree.tostring(tree, encoding="utf-8"))
                    else:
                        docx.writestr(item.filename, original.read(item.filename))
        
        logger.info(f"Document translated successfully: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error translating document: {str(e)}")
        raise Exception(f"Ошибка при переводе документа: {str(e)}")

def clean_temp_files():
    """Clean up temporary files"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            logger.info("Temporary files cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning temporary files: {str(e)}")

def main():
    try:
        st.title("Переводчик документов Tilmash")
        
        # Check for HF_TOKEN in environment
        if 'HF_TOKEN' not in os.environ:
            st.warning("⚠️ Hugging Face токен не установлен. Пожалуйста, установите HF_TOKEN в .env файле.")
        
        # Load translation model
        tilmash_pipeline = load_translation_model()
        
        # Language selection
        st.sidebar.header("Настройки перевода")
        source_lang = st.sidebar.selectbox(
            "Исходный язык", 
            ["Русский", "Казахский"], 
            index=0
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Загрузите документ для перевода", 
            type=['pdf', 'docx']
        )
        
        if uploaded_file is not None:
            # Clean temp files before processing new file
            clean_temp_files()
            
            # Determine source and target languages
            is_rus_to_kaz = source_lang == "Русский"
            source = 'rus' if is_rus_to_kaz else 'kaz'
            target = 'kaz' if is_rus_to_kaz else 'rus'
            
            # Save uploaded file to temp directory
            input_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Prepare output filename
            output_filename = f"translated_{uploaded_file.name}"
            output_path = os.path.join(TEMP_DIR, output_filename)
            
            # Translation button
            if st.button("Перевести документ"):
                try:
                    with st.spinner("Обработка документа..."):
                        # Handle PDF conversion if needed
                        if uploaded_file.name.lower().endswith('.pdf'):
                            temp_docx = os.path.join(TEMP_DIR, "temp_converted.docx")
                            input_path = pdf_to_docx(input_path, temp_docx)
                        
                        # Translate the document
                        translated_path = translate_docx(
                            tilmash_pipeline, 
                            input_path, 
                            output_path, 
                            source, 
                            target
                        )
                        
                        # Provide download button
                        with open(translated_path, "rb") as file:
                            st.download_button(
                                label="Скачать переведенный документ",
                                data=file.read(),
                                file_name=output_filename,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        
                        st.success("Документ успешно переведен!")
                
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
                    logger.error(f"Error in translation process: {str(e)}")
                
                finally:
                    # Clean up temp files after processing
                    clean_temp_files()

    except Exception as e:
        st.error(f"Критическая ошибка: {str(e)}")
        logger.error(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()