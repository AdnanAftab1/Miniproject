from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

df = pd.read_csv("dataset.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
documents = []
ids = []

for i, row in df.iterrows():
    document = Document(
        page_content=(
            f"Activity: {row['Activity']}. "
            f"Average CO2 emission: {row['Avg_CO2_Emission(kg/day)']} kg/day. "
            f"Category: {row['Category']}."
        ),
        metadata={
            "activity": row["Activity"],
            "avg_co2_emission_kg_per_day": float(row["Avg_CO2_Emission(kg/day)"]),
            "category": row["Category"],
        },
        id=str(i)
    )
    ids.append(str(i))
    documents.append(document)
        
vector_store = Chroma(
    collection_name="co2_emissions_dataset",
    persist_directory=db_location,
    embedding_function=embeddings
)
add_documents = vector_store._collection.count() == 0

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

