import os
from dotenv import load_dotenv


# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma



# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()



# =====================================================
# PATHS
# =====================================================

CITY_FOLDER = r"C:\learning\cityrag\city"

VECTOR_DATABASE = r"C:\learning\cityrag\data\example1"



# Create vector folder if not exists

os.makedirs(
    VECTOR_DATABASE,
    exist_ok=True
)



# =====================================================
# LOAD CITY DOCUMENTS
# =====================================================

print("\nLoading city documents...")


loader = DirectoryLoader(

    CITY_FOLDER,

    glob="*.txt",

    loader_cls=TextLoader

)


documents = loader.load()



print(
    "Total city files:",
    len(documents)
)



if len(documents) == 0:

    raise Exception(
        "No .txt files found in city folder"
    )



# =====================================================
# ADD CITY METADATA
# =====================================================

print("\nAdding city metadata...")


for doc in documents:


    file_path = doc.metadata["source"]


    file_name = os.path.basename(
        file_path
    )


    city_name = file_name.replace(
        ".txt",
        ""
    )


    doc.metadata["city"] = city_name



print("City metadata added")



# =====================================================
# SPLIT DOCUMENTS
# =====================================================

print("\nSplitting documents...")


splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200

)



chunks = splitter.split_documents(

    documents

)



print(

    "Total chunks created:",

    len(chunks)

)



# =====================================================
# ADD CHUNK ID
# =====================================================

print("\nAdding chunk IDs...")


for index, chunk in enumerate(chunks):


    chunk.metadata["chunk-id"] = str(index)



print("Chunk metadata added")



# =====================================================
# GEMINI EMBEDDINGS
# =====================================================

print("\nCreating Gemini embeddings...")


api_key = os.getenv(
    "GOOGLE_API_KEY"
)



if not api_key:

    raise Exception(
        "GOOGLE_API_KEY missing in .env file"
    )



embedding_model = GoogleGenerativeAIEmbeddings(

    model="gemini-embedding-001",

    google_api_key=api_key

)


# =====================================================
# CREATE CHROMA VECTOR DATABASE
# =====================================================

print("\nSaving vectors into Chroma...")


vector_db = Chroma(

    collection_name="city_database",

    embedding_function=embedding_model,

    persist_directory=VECTOR_DATABASE

)



# =====================================================
# STORE DOCUMENT CHUNKS
# =====================================================

vector_db.add_documents(

    chunks

)



print("\n===================================")

print("Vector Database Created Successfully")

print("Location:")

print(VECTOR_DATABASE)

print("===================================")



# =====================================================
# TEST SEARCH
# =====================================================


question = "Tell me about Hyderabad"



print("\nSearching:")

print(question)



results = vector_db.similarity_search(

    question,

    k=3

)



print("\nRetrieved Results")



for result in results:


    print("\n--------------------------")


    print(

        result.page_content[:300]

    )


    print("\nMetadata:")


    print(

        result.metadata

    )