from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from pathlib import Path

CHROMA_PATH = "chroma_db"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_context(query: str, k: int = 3) -> list:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return docs

def explain_vessel(mmsi: str, flags: dict) -> str:
    flag_descriptions = []
    if flags.get("fake_mmsi"):
        flag_descriptions.append(f"MMSI {mmsi} is in the reserved/fake MMSI list")
    if flags.get("aivdo_unknown"):
        flag_descriptions.append("Broadcasting AIVDO with Unknown country")
    if flags.get("heading_cog_mismatch"):
        flag_descriptions.append("Heading and COG differ by more than 45 degrees")
    if flags.get("anchored_moving"):
        flag_descriptions.append("Vessel is anchored/moored but reporting movement")
    if flags.get("position_jump"):
        flag_descriptions.append("Vessel position jumped impossibly large distance")

    if not flag_descriptions:
        return "No spoofing indicators detected for this vessel."

    flags_text = "\n".join(f"- {f}" for f in flag_descriptions)

    query = f"AIS spoofing indicators: {', '.join(flag_descriptions)}"
    docs = retrieve_context(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))

    explanation = f"""
VESSEL SPOOFING ANALYSIS — MMSI {mmsi}
{'='*50}

DETECTED FLAGS:
{flags_text}

REGULATORY CONTEXT (from ITU-R M.1371 & IMO SN/Circ.227):
{context[:1500]}

SOURCES: {', '.join(sources)}

CONCLUSION:
This vessel has been flagged as suspicious based on {len(flag_descriptions)} 
spoofing indicator(s). The flags detected are consistent with known AIS 
spoofing patterns documented in maritime regulations.
"""
    return explanation

if __name__ == "__main__":
    print("Testing RAG retriever...")
    print("\n--- Test 1: Simple context retrieval ---")
    docs = retrieve_context("fake MMSI spoofing detection", k=2)
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1} (from {doc.metadata.get('source', 'unknown')}):")
        print(doc.page_content[:200])

    print("\n--- Test 2: Vessel explanation ---")
    explanation = explain_vessel("987654321", {
        "fake_mmsi": True,
        "aivdo_unknown": True,
        "heading_cog_mismatch": False,
        "anchored_moving": False,
        "position_jump": False
    })
    print(explanation)