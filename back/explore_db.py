#!/usr/bin/env python3
"""
Script para explorar la base de datos vectorial de ChromaDB
"""
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from src.rag_agent import _embeddings

# Configuración
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = Path(os.environ.get("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = "arquia"


def main():
    print("=" * 70)
    print("EXPLORADOR DE BASE DE DATOS VECTORIAL - ChromaDB")
    print("=" * 70)
    print(f"\n📁 Ubicación: {PERSIST_DIR}")
    
    # Cargar la base de datos
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )
    
    # Obtener colección
    collection = vectordb._collection
    
    # Estadísticas básicas
    count = collection.count()
    print(f"\n📊 Total de documentos en la colección '{COLLECTION_NAME}': {count}")
    
    if count == 0:
        print("\n⚠️  La base de datos está vacía.")
        return
    
    # Obtener algunos documentos de muestra
    print("\n" + "=" * 70)
    print("MUESTRA DE DOCUMENTOS (primeros 5)")
    print("=" * 70)
    
    results = collection.get(
        limit=5,
        include=["documents", "metadatas"]
    )
    
    for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"]), 1):
        print(f"\n--- Documento {i} ---")
        print(f"Fuente: {metadata.get('source_title', 'N/A')}")
        print(f"Archivo: {metadata.get('title', 'N/A')}")
        print(f"Página: {metadata.get('page', 'N/A')}")
        print(f"Contenido (primeros 200 caracteres):")
        print(f"  {doc[:200]}...")
    
    # Estadísticas por fuente
    print("\n" + "=" * 70)
    print("DOCUMENTOS POR FUENTE")
    print("=" * 70)
    
    all_results = collection.get(include=["metadatas"])
    sources = {}
    for metadata in all_results["metadatas"]:
        source = metadata.get("source_title", "Desconocido")
        sources[source] = sources.get(source, 0) + 1
    
    for source, count in sorted(sources.items()):
        print(f"  • {source}: {count} chunks")
    
    # Búsqueda de ejemplo
    print("\n" + "=" * 70)
    print("EJEMPLO DE BÚSQUEDA SEMÁNTICA")
    print("=" * 70)
    
    query = "What is software architecture?"
    print(f"\n🔍 Query: '{query}'")
    
    results = vectordb.similarity_search(query, k=3)
    
    print(f"\n📋 Top 3 resultados más relevantes:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Resultado {i} ---")
        print(f"Fuente: {doc.metadata.get('source_title', 'N/A')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print(f"Contenido (primeros 300 caracteres):")
        print(f"  {doc.page_content[:300]}...")
        print()


if __name__ == "__main__":
    main()
