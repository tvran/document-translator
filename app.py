import os
import streamlit as st
import zipfile
from lxml import etree
from pdf2docx import Converter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

st.set_option('server.fileWatcherType', 'none')

# Load environment variables
load_dotenv()

@st.cache_resource
def load_translation_model():
    """Load Tilmash translation model with Hugging Face token"""
    # Get Hugging Face token from environment variable
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
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        st.stop()

def pdf_to_docx(pdf_path, docx_path):
    """Convert PDF to DOCX"""
    cv = Converter(pdf_path)
    cv.convert(docx_path, start=0, end=None, continuous=True)
    cv.close()
    return docx_path

def translate_text(tilmash_pipeline, text):
    """Translate text using Tilmash model"""
    return tilmash_pipeline(text)[0]['translation_text'] if text.strip() else text

def translate_docx(tilmash_pipeline, docx_path, output_path, source_lang='rus', target_lang='kaz'):
    """Translate DOCX file while preserving formatting"""
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
    
    # Translate only text within <w:t> tags without breaking styles
    for elem in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
        original_text = elem.text
        if original_text and len(original_text.strip()) > 0:
            elem.text = translate_text(tilmash_pipeline, original_text)
    
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
    
    # Check for HF_TOKEN in environment
    if 'HF_TOKEN' not in os.environ:
        st.warning("⚠️Hugging Face токен не установлен. Пожалуйста, установите HF_TOKEN в .env файле.")
    
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
        # Determine source and target languages
        is_rus_to_kaz = source_lang == "Русский"
        source = 'rus' if is_rus_to_kaz else 'kaz'
        target = 'kaz' if is_rus_to_kaz else 'rus'
        
        # Save uploaded file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Prepare output filename
        output_filename = f"translated_{uploaded_file.name}"
        
        # Convert PDF to DOCX if needed
        if uploaded_file.name.lower().endswith('.pdf'):
            temp_docx = "temp_converted.docx"
            pdf_to_docx(uploaded_file.name, temp_docx)
            input_file = temp_docx
        else:
            input_file = uploaded_file.name
        
        # Translation button
        if st.button("Перевести документ"):
            try:
                # Translate the document
                translated_path = translate_docx(
                    tilmash_pipeline, 
                    input_file, 
                    output_filename, 
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
                
                # Clean up temporary files
                if uploaded_file.name.lower().endswith('.pdf'):
                    os.remove("temp_converted.docx")
                os.remove(uploaded_file.name)
                
            except Exception as e:
                st.error(f"Ошибка при переводе: {str(e)}")

if __name__ == "__main__":
    main()