# ArquIA back


ARCHIA – ARRANQUE LOCAL DESPUÉS DE APAGAR EL COMPUTADOR
(Sistema Windows)

REQUISITOS (ya instalados):

Python 3.11

Node.js + npm

Repositorio clonado

Entorno virtual .venv ya creado

Dependencias ya instaladas

API Key de OpenAI

PASO 1 – ABRIR TERMINALES

Abrir DOS terminales:

Terminal 1: Backend (PowerShell o CMD)
Terminal 2: Frontend (PowerShell o CMD)

PASO 2 – BACKEND: IR AL DIRECTORIO

En la Terminal 1 ejecutar:

cd C:\Users\TU_USUARIO\Documents\GargoyleArchitecture\ArchIA\back

PASO 3 – BACKEND: ACTIVAR EL ENTORNO VIRTUAL

Si usas PowerShell:

..venv\Scripts\Activate.ps1

Si usas CMD:

.venv\Scripts\activate.bat

Confirmación esperada en la terminal:
(.venv)

PASO 4 – BACKEND: VERIFICAR PYTHON

python -V

Debe mostrar:
Python 3.11.x

PASO 5 – BACKEND: DEFINIR API KEY (OBLIGATORIO CADA VEZ)

Si usas PowerShell:

$env:OPENAI_API_KEY="TU_API_KEY_AQUI"

Si usas CMD:

set OPENAI_API_KEY=TU_API_KEY_AQUI

NOTA:
Este paso SIEMPRE se debe hacer después de reiniciar el computador.

PASO 6 – BACKEND: LEVANTAR EL SERVIDOR

python -m uvicorn src.main:app --port 8000

Backend disponible en:
http://localhost:8000

NO cerrar esta terminal mientras se use la aplicación.

PASO 7 – FRONTEND: IR AL DIRECTORIO

En la Terminal 2 ejecutar:

cd C:\Users\TU_USUARIO\Documents\GargoyleArchitecture\ArchIA\front

PASO 8 – FRONTEND: LEVANTAR LA APLICACIÓN

npm run dev

El frontend mostrará una URL similar a:
http://localhost:5173

NO cerrar esta terminal mientras se use la aplicación.

PASO 9 – USAR LA APLICACIÓN

Abrir un navegador web

Entrar a la URL del frontend

Interactuar con la aplicación

El frontend se comunica con el backend en:
http://localhost:8000

PASO 10 – APAGAR EL SISTEMA

En CADA terminal presionar:

Ctrl + C

Esto apaga backend y frontend correctamente.

PASO 11 – COSAS QUE NO SE REPITEN

NO volver a hacer:

Crear el entorno virtual (.venv)

Instalar dependencias con pip

Ejecutar build_vectorstore.py (salvo cambios en documentos)

Crear carpetas docs o feedback_db

Crear el archivo front/.env.local

RESUMEN ULTRA RÁPIDO

Activar .venv

Definir OPENAI_API_KEY

Levantar backend

Levantar frontend

Si estos cuatro pasos funcionan, ArchIA está corriendo correctamente.
