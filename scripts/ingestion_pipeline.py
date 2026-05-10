import os
import glob
import pandas as pd
import chromadb
import nest_asyncio
from dotenv import load_dotenv

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import PyMuPDFReader

nest_asyncio.apply()
load_dotenv()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
pdf_reader = PyMuPDFReader()

splitter = SemanticSplitterNodeParser(
    buffer_size=1, 
    breakpoint_percentile_threshold=95, 
    embed_model=embed_model
)

def load_csv_documents():
    csv_path = "./data/product_catalog_clean.csv"
    if not os.path.exists(csv_path):
        return []
        
    df = pd.read_csv(csv_path)
    csv_docs = []
    
    for _, row in df.iterrows():
        content = str(row['LLM_Context'])
        metadata = {
            "source_type": "product_catalog",
            "product_code": str(row['Product Code']),
            "brand": str(row['Brand'])
        }
        doc = Document(text=content, metadata=metadata)
        csv_docs.append(doc)
        
    return csv_docs

def run_ingestion():
    # Đọc PDF bằng PyMuPDF (local, nhanh)
    pdf_files = glob.glob("./data/Policies/*.pdf")
    pdf_documents = []
    
    print("📄 Đang đọc PDF files...")
    for pdf_path in pdf_files:
        print(f"  - {os.path.basename(pdf_path)}")
        docs = pdf_reader.load_data(pdf_path)
        # Đánh dấu là PDF
        for doc in docs:
            doc.metadata["source_type"] = "policy_pdf"
        pdf_documents.extend(docs)
    
    print(f"✅ Đọc xong {len(pdf_documents)} documents từ PDF")
    
    csv_documents = load_csv_documents()
    print(f"✅ Đọc xong {len(csv_documents)} documents từ CSV")

    # Bước 1: Hierarchical Parsing CHỈ cho PDF
    print("\n🔨 Bước 1: Hierarchical Parsing (CHỈ PDF)...")
    hierarchical_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128]
    )
    pdf_nodes = hierarchical_parser.get_nodes_from_documents(pdf_documents)
    pdf_leaf_nodes = get_leaf_nodes(pdf_nodes)
    pdf_parent_nodes = [node for node in pdf_nodes if node not in pdf_leaf_nodes]
    
    print(f"✅ PDF: {len(pdf_nodes)} nodes ({len(pdf_parent_nodes)} parent + {len(pdf_leaf_nodes)} leaf)")
    
    # Bước 2: Semantic Chunking CHỈ cho PDF leaf nodes
    print("\n🔨 Bước 2: Semantic Chunking (CHỈ PDF leaf nodes)...")
    semantic_pdf_nodes = []
    for i, leaf in enumerate(pdf_leaf_nodes):
        if i % 5 == 0:
            print(f"  Processing {i}/{len(pdf_leaf_nodes)}...")
        temp_doc = Document(text=leaf.text, metadata=leaf.metadata)
        split_nodes = splitter.get_nodes_from_documents([temp_doc])
        
        for node in split_nodes:
            node.relationships = leaf.relationships
            semantic_pdf_nodes.append(node)
    
    print(f"✅ Semantic split: {len(semantic_pdf_nodes)} PDF nodes")
    
    # Bước 3: CSV nodes - KHÔNG cần hierarchical/semantic (đã clean)
    print("\n🔨 Bước 3: Xử lý CSV (simple chunking)...")
    from llama_index.core.node_parser import SentenceSplitter
    simple_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    csv_nodes = simple_splitter.get_nodes_from_documents(csv_documents)
    print(f"✅ CSV: {len(csv_nodes)} nodes")
    
    # Kết hợp: PDF parent + PDF semantic leaf + CSV simple
    all_leaf_nodes = semantic_pdf_nodes + csv_nodes
    
    # Bước 4: Setup ChromaDB
    print("\n🔨 Bước 4: Lưu vào ChromaDB...")
    db = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "sales_copilot_vdb"
    
    try:
        db.delete_collection(collection_name)
    except Exception:
        pass 
        
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Bước 5: Setup Docstore
    docstore = SimpleDocumentStore()
    docstore.add_documents(pdf_parent_nodes + all_leaf_nodes)
    
    # Bước 6: Tạo StorageContext
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )
    
    # Bước 7: Tạo Vector Index (CHỈ leaf nodes)
    print("\n🔨 Bước 5: Embedding leaf nodes...")
    index = VectorStoreIndex(
        all_leaf_nodes,
        storage_context=storage_context, 
        embed_model=embed_model,
        show_progress=True 
    )
    
    # Bước 8: Persist Docstore
    docstore.persist(persist_path="./chroma_db/docstore.json")
    
    print(f"\n✅ Ingestion hoàn tất!")
    print(f"   📦 PDF Parent nodes: {len(pdf_parent_nodes)}")
    print(f"   🔍 PDF Leaf nodes (semantic): {len(semantic_pdf_nodes)}")
    print(f"   � CSV nodes (simple): {len(csv_nodes)}")
    print(f"   💾 Tổng trong Vector DB: {len(all_leaf_nodes)}")

if __name__ == "__main__":
    run_ingestion()
