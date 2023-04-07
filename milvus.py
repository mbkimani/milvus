from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Milvus


from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import gradio as gr
# loader = TextLoader('../../../state_of_the_union.txt')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()

# start App implementation
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="client-side")

@app.get('/', response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.get('/favicon.ico')
async def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "static")
    return FileResponse(path=file_path, headers={"Content-Disposition": "attachment; filename=" + file_name})

# end App implementation

import os
from environs import Env

#get OpenAIKey from .env
env = Env()
env.read_env()

OPENAI_API_KEY = os.getenv('OPENAI_API')
model_name = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(document_model_name=model_name, query_model_name=model_name, openai_api_key=OPENAI_API_KEY)

loaderPdf = UnstructuredPDFLoader("./Designing_the_Internet_of_Things.pdf")
data = loaderPdf.load()

print(f"You have {len(data)} documents in your data")
print(f"There are {len(data[0].page_content)} characters in your document")

#chunk your data into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
texts = text_splitter.split_documents(data)

print(f"Now you have {len(texts)} documents.")

vector_db = Milvus.from_texts(
    [t.page_content for t in texts],
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
# q and a and relevant docs
# query = "What is a business model canvas"
# docs = vector_db.similarity_search(query, include_metadata=True)

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

# print(chain.run(input_documents=docs, question=query))

def qanda(userquestion):
    query = userquestion
    docs = vector_db.similarity_search(query, include_metadata=True)
    return chain.run(input_documents=docs, question=query)

webInterface = gr.Interface(fn=qanda,
                             inputs=gr.inputs.Textbox(lines=7, label="Enter your question about the document"),
                             outputs="text",
                             title="Just in Time Chatbot")

#io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, webInterface, path='/gradio')
