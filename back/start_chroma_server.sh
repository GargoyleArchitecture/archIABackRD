#!/bin/bash
# Script para levantar ChromaDB como servidor HTTP

echo "🚀 Levantando ChromaDB Server..."
echo "📁 Base de datos: ./chroma_db"
echo "🌐 URL: http://localhost:8001"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo "================================================"

cd "$(dirname "$0")"
export CHROMA_DB_IMPL=duckdb+parquet
export PERSIST_DIRECTORY=./chroma_db

# Intenta ejecutar el servidor de chroma
chroma run --path ./chroma_db --host localhost --port 8001
