#!/usr/bin/env python3
"""
Servidor HTTP para ChromaDB con interfaz web simple
Levanta el servidor en http://localhost:8001
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
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import uvicorn

# Configuración
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = Path(os.environ.get("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = "arquia"

app = FastAPI(title="ChromaDB Explorer", version="1.0.0")

# Cargar la base de datos
vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings(),
    persist_directory=str(PERSIST_DIR),
)
collection = vectordb._collection


@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal con interfaz web"""
    count = collection.count()
    
    # Obtener estadísticas
    all_results = collection.get(include=["metadatas", "documents"])
    sources = {}
    for metadata in all_results["metadatas"]:
        source = metadata.get("source_title", "Desconocido")
        sources[source] = sources.get(source, 0) + 1
    
    sources_html = "".join([f"<li><strong>{src}</strong>: {cnt} chunks</li>" 
                           for src, cnt in sorted(sources.items())])
    
    # Obtener lista de documentos (primeros 50 para no sobrecargar)
    docs_sample = collection.get(limit=50, include=["documents", "metadatas"])
    documents_html = ""
    for i, (doc, meta) in enumerate(zip(docs_sample["documents"], docs_sample["metadatas"]), 1):
        content_preview = doc[:200].replace("\n", " ") if doc else ""
        documents_html += f"""
        <div class="doc-item">
            <div class="doc-header">
                <span class="doc-number">#{i}</span>
                <span class="doc-source">{meta.get('source_title', 'N/A')}</span>
                <span class="doc-page">Página {meta.get('page', 'N/A')}</span>
            </div>
            <div class="doc-content">{content_preview}...</div>
        </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ChromaDB Explorer - ArchIA</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-card h3 {{
                margin: 0 0 10px 0;
                color: #667eea;
                font-size: 1em;
                text-transform: uppercase;
            }}
            .stat-card .value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #333;
            }}
            .search-section {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}
            .search-section h2 {{
                margin: 0 0 20px 0;
                color: #333;
            }}
            .search-form {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }}
            input[type="text"] {{
                flex: 1;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 16px;
            }}
            input[type="text"]:focus {{
                outline: none;
                border-color: #667eea;
            }}
            input[type="number"] {{
                width: 80px;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 16px;
            }}
            button {{
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
            }}
            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .sources {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .sources h2 {{
                margin: 0 0 20px 0;
                color: #333;
            }}
            .sources ul {{
                list-style: none;
                padding: 0;
            }}
            .sources li {{
                padding: 10px 0;
                border-bottom: 1px solid #e0e0e0;
            }}
            .sources li:last-child {{
                border-bottom: none;
            }}
            .documents-list {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-top: 30px;
                max-height: 800px;
                overflow-y: auto;
            }}
            .documents-list h2 {{
                margin: 0 0 20px 0;
                color: #333;
                position: sticky;
                top: 0;
                background: white;
                padding-bottom: 15px;
                z-index: 10;
            }}
            .doc-item {{
                background: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 12px;
                border-left: 3px solid #667eea;
                transition: all 0.2s;
            }}
            .doc-item:hover {{
                background: #f0f0f0;
                border-left-color: #764ba2;
                transform: translateX(5px);
            }}
            .doc-header {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 10px;
                flex-wrap: wrap;
            }}
            .doc-number {{
                background: #667eea;
                color: white;
                padding: 4px 10px;
                border-radius: 15px;
                font-size: 0.85em;
                font-weight: bold;
            }}
            .doc-source {{
                color: #667eea;
                font-weight: bold;
                flex: 1;
            }}
            .doc-page {{
                color: #999;
                font-size: 0.9em;
            }}
            .doc-content {{
                color: #555;
                line-height: 1.5;
                font-size: 0.95em;
            }}
            .load-more {{
                text-align: center;
                margin-top: 20px;
            }}
            .load-more button {{
                background: #667eea;
                padding: 10px 25px;
            }}
            #results {{
                margin-top: 20px;
            }}
            .result-item {{
                background: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #667eea;
            }}
            .result-item h4 {{
                margin: 0 0 10px 0;
                color: #667eea;
            }}
            .result-item .metadata {{
                color: #666;
                font-size: 0.9em;
                margin-bottom: 10px;
            }}
            .result-item .content {{
                line-height: 1.6;
                color: #333;
            }}
            .api-docs {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-top: 30px;
            }}
            .api-docs h3 {{
                margin: 0 0 15px 0;
                color: #333;
            }}
            .api-docs a {{
                color: #667eea;
                text-decoration: none;
                font-weight: bold;
            }}
            .api-docs a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 ChromaDB Explorer</h1>
            <p>Base de datos vectorial para ArchIA - Sistema RAG de Arquitectura de Software</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Documentos</h3>
                <div class="value">{count}</div>
            </div>
            <div class="stat-card">
                <h3>Colección</h3>
                <div class="value" style="font-size: 1.5em;">{COLLECTION_NAME}</div>
            </div>
            <div class="stat-card">
                <h3>Fuentes</h3>
                <div class="value">{len(sources)}</div>
            </div>
        </div>
        
        <div class="search-section">
            <h2>🔎 Búsqueda Semántica</h2>
            <form class="search-form" onsubmit="search(event)">
                <input type="text" id="query" placeholder="Escribe tu consulta aquí... (e.g., 'software architecture patterns')" required>
                <input type="number" id="k" value="5" min="1" max="20" title="Número de resultados">
                <button type="submit">Buscar</button>
            </form>
            <div id="results"></div>
        </div>
        
        <div class="sources">
            <h2>📚 Documentos por Fuente</h2>
            <ul>
                {sources_html}
            </ul>
        </div>
        
        <div class="documents-list">
            <h2>📄 Lista de Documentos (Primeros 50)</h2>
            <div id="documents-container">
                {documents_html}
            </div>
            <div class="load-more">
                <button onclick="loadMoreDocuments()">Cargar más documentos</button>
            </div>
        </div>
        
        <div class="api-docs">
            <h3>📖 Documentación API</h3>
            <p>
                Accede a la documentación interactiva de la API:
                <a href="/docs" target="_blank">Swagger UI</a> | 
                <a href="/redoc" target="_blank">ReDoc</a>
            </p>
        </div>
        
        <script>
            let currentOffset = 50;
            
            async function search(event) {{
                event.preventDefault();
                const query = document.getElementById('query').value;
                const k = document.getElementById('k').value;
                const resultsDiv = document.getElementById('results');
                
                resultsDiv.innerHTML = '<p>🔄 Buscando...</p>';
                
                try {{
                    const response = await fetch(`/search?query=${{encodeURIComponent(query)}}&k=${{k}}`);
                    const data = await response.json();
                    
                    if (data.results.length === 0) {{
                        resultsDiv.innerHTML = '<p>❌ No se encontraron resultados.</p>';
                        return;
                    }}
                    
                    let html = `<h3>✅ Encontrados ${{data.results.length}} resultados:</h3>`;
                    data.results.forEach((result, index) => {{
                        html += `
                            <div class="result-item">
                                <h4>Resultado ${{index + 1}}</h4>
                                <div class="metadata">
                                    📖 <strong>${{result.metadata.source_title}}</strong> | 
                                    📄 Página: ${{result.metadata.page}}
                                </div>
                                <div class="content">${{result.content}}</div>
                            </div>
                        `;
                    }});
                    
                    resultsDiv.innerHTML = html;
                }} catch (error) {{
                    resultsDiv.innerHTML = '<p>❌ Error al realizar la búsqueda: ' + error + '</p>';
                }}
            }}
            
            async function loadMoreDocuments() {{
                const container = document.getElementById('documents-container');
                const button = event.target;
                button.textContent = '⏳ Cargando...';
                button.disabled = true;
                
                try {{
                    const response = await fetch(`/documents?limit=50&offset=${{currentOffset}}`);
                    const data = await response.json();
                    
                    if (data.documents.length === 0) {{
                        button.textContent = 'No hay más documentos';
                        return;
                    }}
                    
                    data.documents.forEach((doc, index) => {{
                        const docNum = currentOffset + index + 1;
                        const contentPreview = doc.content.substring(0, 200).replace(/\\n/g, ' ');
                        const docHtml = `
                            <div class="doc-item">
                                <div class="doc-header">
                                    <span class="doc-number">#${{docNum}}</span>
                                    <span class="doc-source">${{doc.metadata.source_title || 'N/A'}}</span>
                                    <span class="doc-page">Página ${{doc.metadata.page || 'N/A'}}</span>
                                </div>
                                <div class="doc-content">${{contentPreview}}...</div>
                            </div>
                        `;
                        container.insertAdjacentHTML('beforeend', docHtml);
                    }});
                    
                    currentOffset += data.documents.length;
                    button.textContent = 'Cargar más documentos';
                    button.disabled = false;
                    
                }} catch (error) {{
                    button.textContent = 'Error al cargar';
                    console.error('Error:', error);
                }}
            }}
        </script>
    </body>
    </html>
    """


@app.get("/api/stats")
async def get_stats():
    """Obtener estadísticas de la base de datos"""
    count = collection.count()
    all_results = collection.get(include=["metadatas"])
    
    sources = {}
    for metadata in all_results["metadatas"]:
        source = metadata.get("source_title", "Desconocido")
        sources[source] = sources.get(source, 0) + 1
    
    return {
        "total_documents": count,
        "collection_name": COLLECTION_NAME,
        "sources": sources,
        "persist_directory": str(PERSIST_DIR)
    }


@app.get("/search")
async def search(
    query: str = Query(..., description="Consulta de búsqueda"),
    k: int = Query(5, ge=1, le=20, description="Número de resultados")
):
    """Realizar búsqueda semántica"""
    results = vectordb.similarity_search(query, k=k)
    
    return {
        "query": query,
        "k": k,
        "total_results": len(results),
        "results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    }


@app.get("/documents")
async def get_documents(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Obtener documentos con paginación"""
    results = collection.get(
        limit=limit,
        offset=offset,
        include=["documents", "metadatas"]
    )
    
    return {
        "limit": limit,
        "offset": offset,
        "documents": [
            {
                "content": doc,
                "metadata": meta
            }
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    }


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ChromaDB Explorer Server")
    print("=" * 70)
    print(f"📁 Base de datos: {PERSIST_DIR}")
    print(f"📊 Total documentos: {collection.count()}")
    print(f"🌐 Interfaz web: http://localhost:8001")
    print(f"📖 API docs: http://localhost:8001/docs")
    print("=" * 70)
    print("\nPresiona Ctrl+C para detener el servidor\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
