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

3. Create .env file and add Llama Cloud and Google API Keys
```
GOOGLE_API_KEY="your_api_key"
LLAMA_CLOUD_API_KEY = "your_api_key"
```

4. Run the notebook

5. Run app.py
```
uvicorn app:app --reload
```

6. Run frontend
 ```
 npm run dev
 ```

