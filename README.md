# RAG for User Manuals

## Getting Started
1. Install uv
```
pip install uv
uv venv --python 3.12.12
.venv/Scripts/activate
```

2. Install necessary dependencies
```
uv sync
```

3. Create .env file and add API Keys
```
APP_ENV=dev
GOOGLE_API_KEY=
LLAMA_CLOUD_API_KEY=
GROQ_API_KEY=
COHERE_API_KEY=
```

4. Run app.py
```
uvicorn app:app --reload
```

6. Run frontend
 ```
 npm run dev
 ```
