from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


from pathlib import Path
 

pdf_path = Path(__file__).parent / "nodejs.pdf"

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load() 
# this gives all the content of the PDF page wise in docs so that you can access it using eg docs[3], docs[12] etc

print(docs[12])
 
# split the docs into smaller chunks having overlapping of 400 so that it can understand the context

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(docs)

#Vector Embedding of this chunks and store it into database(qdrant we are using)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="Learning_RAG"
)

print("Indexding of documents done")
