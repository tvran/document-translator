import os
import streamlit as st
import zipfile
from lxml import etree
from pdf2docx import Converter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import traceback
import sys

# Streamlit configuration to avoid PyTorch error
st.set_option('server.fileWatcherType', 'none')

# Load environment variables
load_dotenv()

def print_debug(message):
    """Print debug message to both console and streamlit"""
    print(message, file=sys.stderr)
    st.write(message)

@st.cache_resource
def load_translation_model():
    """Load Tilmash translation model with Hugging Face token"""
    print_debug("📚 Начинаем загрузку модели...")
    try:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print_debug("❌ HF_TOKEN не найден в .env")
            st.error("Hugging Face токен не найден. Пожалуйста, установите HF_TOKEN в .env файле.")
            st.stop()
        
        print_debug("🔄 Загружаем модель...")
        model = AutoModelForSeq2SeqLM.from_pretrained("issai/tilmash", token=hf_token)
        print_debug("🔄 Загружаем токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained("issai/tilmash", token=hf_token)
        print_debug("✅ Модель успешно загружена")
        return pipeline("translation", model=model, tokenizer=tokenizer, 
                        src_lang="rus_Cyrl", tgt_lang="kaz_Cyrl", max_length=1000)
    except Exception as e:
        print_debug(f"❌ Ошибка при загрузке модели: {str(e)}")
        print_debug(f"Traceback: {traceback.format_exc()}")
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        st.stop()

def pdf_to_docx(pdf_path, docx_path):
    """Convert PDF to DOCX"""
    print_debug(f"📄 Начинаем конвертацию PDF: {pdf_path} -> {docx_path}")
    try:
        cv = Converter(pdf_path)
        print_debug("🔄 Конвертация...")
        cv.convert(docx_path, start=0, end=None, continuous=True)
        cv.close()
        if os.path.exists(docx_path):
            print_debug(f"✅ PDF успешно сконвертирован в {docx_path}")
            return docx_path
        else:
            print_debug("❌ Файл DOCX не был создан")
            raise Exception("Файл DOCX не был создан")
    except Exception as e:
        print_debug(f"❌ Ошибка при конвертации PDF: {str(e)}")
        print_debug(f"Traceback: {traceback.format_exc()}")
        raise

def translate_text(tilmash_pipeline, text):
    """Translate text using Tilmash model"""
    try:
        if not text.strip():
            return text
        print_debug(f"🔄 Переводим текст: {text[:50]}...")
        result = tilmash_pipeline(text)[0]['translation_text']
        print_debug(f"✅ Перевод выполнен: {result[:50]}...")
        return result
    except Exception as e:
        print_debug(f"❌ Ошибка при переводе текста: {str(e)}")
        print_debug(f"Проблемный текст: {text}")
        print_debug(f"Traceback: {traceback.format_exc()}")
        raise

def translate_docx(tilmash_pipeline, docx_path, output_path, source_lang='rus', target_lang='kaz'):
    """Translate DOCX file while preserving formatting"""
    print_debug(f"📄 Начинаем перевод документа: {docx_path}")
    try:
        # Modify pipeline based on translation direction
        if source_lang == 'kaz':
            tilmash_pipeline.tokenizer.src_lang = "kaz_Cyrl"
            tilmash_pipeline.tokenizer.tgt_lang = "rus_Cyrl"
        else:
            tilmash_pipeline.tokenizer.src_lang = "rus_Cyrl"
            tilmash_pipeline.tokenizer.tgt_lang = "kaz_Cyrl"

        print_debug("🔄 Читаем DOCX файл...")
        with zipfile.ZipFile(docx_path, "r") as docx:
            xml_content = docx.read("word/document.xml")
        
        print_debug("🔄 Парсим XML...")
        tree = etree.fromstring(xml_content)
        
        print_debug("🔄 Переводим содержимое...")
        # Translate only text within <w:t> tags without breaking styles
        for elem in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
            original_text = elem.text
            if original_text and len(original_text.strip()) > 0:
                elem.text = translate_text(tilmash_pipeline, original_text)
        
        print_debug("🔄 Сохраняем переведенный документ...")
        # Save new DOCX
        with zipfile.ZipFile(output_path, "w") as docx:
            for item in zipfile.ZipFile(docx_path, "r").infolist():
                if item.filename == "word/document.xml":
                    docx.writestr(item.filename, etree.tostring(tree, encoding="utf-8"))
                else:
                    docx.writestr(item.filename, zipfile.ZipFile(docx_path, "r").read(item.filename))
        
        print_debug(f"✅ Документ успешно переведен и сохранен: {output_path}")
        return output_path
    except Exception as e:
        print_debug(f"❌ Ошибка при переводе документа: {str(e)}")
        print_debug(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    st.title("Переводчик документов Tilmash")
    
    print_debug("🚀 Приложение запущено")
    
    # Check for HF_TOKEN in environment
    if 'HF_TOKEN' not in os.environ:
        print_debug("⚠️ HF_TOKEN не установлен")
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
        print_debug(f"📁 Загружен файл: {uploaded_file.name}")
        
        # Determine source and target languages
        is_rus_to_kaz = source_lang == "Русский"
        source = 'rus' if is_rus_to_kaz else 'kaz'
        target = 'kaz' if is_rus_to_kaz else 'rus'
        
        try:
            # Save uploaded file
            print_debug("🔄 Сохраняем загруженный файл...")
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Prepare output filename
            output_filename = f"translated_{uploaded_file.name}"
            
            # Convert PDF to DOCX if needed
            if uploaded_file.name.lower().endswith('.pdf'):
                print_debug("🔄 Обнаружен PDF файл, начинаем конвертацию...")
                temp_docx = "temp_converted.docx"
                pdf_to_docx(uploaded_file.name, temp_docx)
                input_file = temp_docx
            else:
                input_file = uploaded_file.name
            
            # Translation button
            if st.button("Перевести документ"):
                print_debug("🔄 Начинаем процесс перевода...")
                try:
                    # Translate the document
                    translated_path = translate_docx(
                        tilmash_pipeline, 
                        input_file, 
                        output_filename, 
                        source, 
                        target
                    )
                    
                    print_debug("🔄 Подготавливаем файл для скачивания...")
                    # Provide download button
                    with open(translated_path, "rb") as file:
                        st.download_button(
                            label="Скачать переведенный документ",
                            data=file.read(),
                            file_name=output_filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    
                    st.success("Документ успешно переведен!")
                    print_debug("✅ Процесс перевода завершен успешно")
                    
                    # Clean up temporary files
                    print_debug("🧹 Очищаем временные файлы...")
                    if uploaded_file.name.lower().endswith('.pdf'):
                        os.remove(temp_docx)
                    os.remove(uploaded_file.name)
                    
                except Exception as e:
                    print_debug(f"❌ Ошибка в процессе перевода: {str(e)}")
                    print_debug(f"Traceback: {traceback.format_exc()}")
                    st.error(f"Ошибка при переводе: {str(e)}")
        
        except Exception as e:
            print_debug(f"❌ Ошибка при обработке файла: {str(e)}")
            print_debug(f"Traceback: {traceback.format_exc()}")
            st.error(f"Ошибка при обработке файла: {str(e)}")

if __name__ == "__main__":
    main()