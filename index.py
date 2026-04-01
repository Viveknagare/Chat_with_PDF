from langchain_community.document_loaders import PyPDFLoader

file_path = ""

loader = PyPDFLoader(file_path)

docs = loader.load() 
# this gives all the content of the PDF page wise in docs so that you can access it using eg docs[3], docs[12] etc

print(docs[0])