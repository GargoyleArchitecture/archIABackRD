#!/bin/bash
# Script simplificado para levantar el servidor web de ChromaDB

echo "🚀 Iniciando ChromaDB Web Explorer..."
echo ""

cd "$(dirname "$0")"

# Ejecutar con el Python del entorno virtual
/Users/santiagocasasbuenasalarcon/Documents/universidad/tesis/archIABack/.venv/bin/python chroma_web.py
