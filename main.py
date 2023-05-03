import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# openai_api_key = "YOUR API KEY"
# or get it as enviremental variable:
openai_api_key = os.environ['OPENAI_API_KEY']

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
chain = load_qa_chain(ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7), chain_type="stuff")

pdf_directory = "data"
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

loaders = []
for pdf_name in pdf_files:
    file_path = "{}/{}".format(pdf_directory, pdf_name)
    loader = PyPDFLoader(file_path)
    loaders.append(loader)

docs = []
for loader in loaders:
    docs.extend(loader.load())
documents = text_splitter.split_documents(docs)
docsearch = FAISS.from_documents(documents, embeddings)

print("Type exit() to quit at any time")

while 1:
    # Here you can ask it everything you want.
    query = input("What is the question? >> ")
    if query == "exit()":
        exit()
    input_documents = docsearch.similarity_search(query)
    message = chain.run(input_documents=input_documents, question=query)
    print(message)
