import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
import shutil
os.environ["GROQ_API_KEY"]="your_api_key"
temp_dir="uploaded_files"
vectorstore_dir="vectorstore"
os.makedirs(temp_dir,exist_ok=True)
os.makedirs(vectorstore_dir,exist_ok=True)
vectorstore=None

def build_vectorstore(file_path):
  global vectorstore
  if os.path.exists(vectorstore_dir):
    shutil.rmtree(vectorstore_dir)
  loader=PyPDFLoader(file_path)
  documents=loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  chunks = text_splitter.split_documents(documents)
  embeddings=HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
  )
  vectorstore=Chroma.from_documents(chunks,embeddings,persist_directory=vectorstore_dir)
  vectorstore.persist()
  return "Vectorestore built successfully. You can now ask questions"

def ask_questions(question):
  global vectorstore
  if not vectorstore:
    return "Please upload a document first"
  retriever=vectorstore.as_retriever()
  qa_chain=RetrievalQA.from_chain_type(
      llm=ChatGroq(model="llama3-8b-8192"),
      retriever=retriever,
  )
  response=qa_chain.run(question)
  return response

def handle_file_upload(file):
  file_path=os.path.join(temp_dir,"uploaded_document1.pdf")
  with open(file_path,"wb") as f:
    f.write(file)
  return build_vectorstore(file_path)

def clear_all():
    """Clear uploaded files and vectorstore."""
    global vectorstore
    vectorstore = None
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(vectorstore_dir):
        shutil.rmtree(vectorstore_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(vectorstore_dir, exist_ok=True)
    return "All data cleared. You can upload a new document."

with gr.Blocks() as chatbot_ui:
    gr.Markdown("# PDF Chatbot with LangChain")

    with gr.Row():
        file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="binary")
        upload_button = gr.Button("Upload and Process")

    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
        ask_button = gr.Button("Ask")

    answer_output = gr.Textbox(label="Answer", interactive=False)

    clear_button = gr.Button("Clear All")

    
    upload_button.click(handle_file_upload, inputs=file_upload, outputs=status_output)
    ask_button.click(ask_questions, inputs=question_input, outputs=answer_output)
    clear_button.click(clear_all, outputs=status_output)

chatbot_ui.launch(debug=True,share=True)
