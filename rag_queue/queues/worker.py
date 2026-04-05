from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="Learning_RAG"
)

def process_query(query:str):
    print("Searching chunks", query)
    search_result = vector_db.similarity_search(query)

    context = "\n\n\n".join([f"Page Content: {result.page_content}\n Page Number: {result.metadata['page_label']} \nFile Location: {result.metadata['source']}" for result in search_result
    ])

    SYSTEM_PROMPT = f"""
    You are Helpful AI Assistant who answers user query based on available      context retrived from PDF file along with page_contents and page number.
    
    you should only answer the user based on the following context and navigate      the user to open the right page number to know more

    Context:
    {context}
    """

    response = openai_client.chat.completions.create(
       model="gpt-5",
       messages=[
           {"role" : "system", "content" : SYSTEM_PROMPT},
           {"role" : "user", "content" : query}
       ]
    )

    print(f"🤖: {response.choices[0].message.content}")
    return response.choices[0].message.content
        




