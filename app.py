import os
import io
import numpy as np
import torch
import open_clip
import faiss
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Semantic Image Search", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_DIR = "coco_images"
INDEX_PATH = "image_index.faiss"
PATHS_PATH = "paths.npy"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading OpenCLIP {MODEL_NAME} ({PRETRAINED}) on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(device).eval()

index = None
paths = []

def load_or_build_index():
    global index, paths
    if os.path.exists(INDEX_PATH) and os.path.exists(PATHS_PATH):
        index = faiss.read_index(INDEX_PATH)
        paths = list(np.load(PATHS_PATH))
        print(f"Loaded existing index with {index.ntotal} vectors.")
    else:
        print("No index found. Initializing empty index.")
        index = faiss.IndexFlatIP(512)
        paths = []

load_or_build_index()

@app.get("/images/{filename}")
def get_image(filename: str):
    filepath = os.path.join(IMG_DIR, filename)
    if not os.path.exists(filepath):
        # Fallback to test_images
        filepath = os.path.join("test_images", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.post("/api/search")
def search_images(query: str = Form(...), top_k: int = Form(6)):
    global index, paths
    if index is None or index.ntotal == 0:
        return {"query": query, "results": []}

    text_tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    text_emb_np = text_emb.cpu().numpy().astype('float32')
    k = min(top_k, index.ntotal)
    scores, indices = index.search(text_emb_np, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        filename = paths[idx]
        results.append({
            "filename": filename,
            "url": f"/images/{filename}",
            "score": round(float(score), 4),
            "percentage": round(max(0.0, float(score)) * 100, 1)
        })

    return {"query": query, "results": results, "total_indexed": index.ntotal}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    global index, paths
    os.makedirs(IMG_DIR, exist_ok=True)
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    filename = file.filename
    filepath = os.path.join(IMG_DIR, filename)
    img.save(filepath)

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb /= emb.norm(dim=-1, keepdim=True)

    emb_np = emb.cpu().numpy().astype('float32')
    
    if index is None:
        index = faiss.IndexFlatIP(512)
        paths = []

    index.add(emb_np)
    paths.append(filename)

    faiss.write_index(index, INDEX_PATH)
    np.save(PATHS_PATH, np.array(paths))

    return {"message": "Image uploaded & indexed successfully", "filename": filename, "total_indexed": index.ntotal}

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Image Search — OpenCLIP & FAISS</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(22, 31, 49, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --text: #f3f4f6;
            --text-sub: #9ca3af;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }

        body {
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.15) 0%, transparent 40%);
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        h1 {
            font-size: 2.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-sub);
            font-size: 1.1rem;
        }

        .stats-badge {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.35rem 1rem;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid var(--accent-glow);
            border-radius: 999px;
            color: #c7d2fe;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .search-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.75rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .search-box {
            display: flex;
            gap: 1rem;
        }

        input[type="text"] {
            flex: 1;
            background: rgba(10, 15, 26, 0.8);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            color: #fff;
            font-size: 1.05rem;
            outline: none;
            transition: all 0.2s ease;
        }

        input[type="text"]:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            padding: 0 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px var(--accent-glow);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--accent-glow);
        }

        .upload-section {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .file-label {
            cursor: pointer;
            color: #a5b4fc;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            transition: all 0.3s ease;
            position: relative;
        }

        .card:hover {
            transform: translateY(-6px);
            border-color: rgba(99, 102, 241, 0.5);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
        }

        .img-container {
            width: 100%;
            height: 240px;
            overflow: hidden;
            background: #000;
        }

        .img-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.4s ease;
        }

        .card:hover .img-container img {
            transform: scale(1.05);
        }

        .card-body {
            padding: 1rem 1.25rem;
        }

        .card-title {
            font-size: 0.95rem;
            font-weight: 500;
            color: #e5e7eb;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 0.5rem;
        }

        .score-bar-bg {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }

        .score-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #ec4899);
            border-radius: 3px;
        }

        .score-text {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-sub);
            margin-top: 0.4rem;
        }

        .loading {
            text-align: center;
            padding: 3rem;
            color: var(--text-sub);
            font-size: 1.1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Semantic Image Search</h1>
            <p class="subtitle">Natural Language Photo Retrieval powered by OpenCLIP & FAISS on GPU</p>
            <div class="stats-badge" id="stats">Indexed Images: Ready</div>
        </header>

        <div class="search-card">
            <form id="search-form" onsubmit="handleSearch(event)">
                <div class="search-box">
                    <input type="text" id="query-input" placeholder="Search photos (e.g. 'a cute dog playing on grass', 'red sports car')..." required>
                    <button type="submit">Search</button>
                </div>
            </form>
            <div class="upload-section">
                <label class="file-label">
                    ➕ Index New Photo:
                    <input type="file" id="file-input" accept="image/*" onchange="handleUpload(event)" style="display:none">
                </label>
                <span id="upload-status" style="font-size:0.9rem; color:#a5b4fc;"></span>
            </div>
        </div>

        <div class="grid" id="results-grid"></div>
    </div>

    <script>
        async function handleSearch(e) {
            e.preventDefault();
            const query = document.getElementById('query-input').value;
            const grid = document.getElementById('results-grid');
            grid.innerHTML = '<div class="loading">Searching vector database...</div>';

            const formData = new FormData();
            formData.append('query', query);
            formData.append('top_k', 8);

            try {
                const res = await fetch('/api/search', { method: 'POST', body: formData });
                const data = await res.json();
                
                document.getElementById('stats').innerText = `Indexed Images: ${data.total_indexed}`;

                if (!data.results || data.results.length === 0) {
                    grid.innerHTML = '<div class="loading">No matching images found.</div>';
                    return;
                }

                grid.innerHTML = data.results.map(item => `
                    <div class="card">
                        <div class="img-container">
                            <img src="${item.url}" alt="${item.filename}" loading="lazy">
                        </div>
                        <div class="card-body">
                            <div class="card-title" title="${item.filename}">${item.filename}</div>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width: ${Math.min(100, Math.max(10, item.score * 300))}%"></div>
                            </div>
                            <div class="score-text">
                                <span>Similarity Score</span>
                                <span>${item.score}</span>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (err) {
                grid.innerHTML = '<div class="loading" style="color:#ef4444">Error connecting to search engine.</div>';
            }
        }

        async function handleUpload(e) {
            const file = e.target.files[0];
            if (!file) return;

            const status = document.getElementById('upload-status');
            status.innerText = "Indexing photo on GPU...";

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                status.innerText = `Uploaded & Indexed: ${data.filename}`;
                document.getElementById('stats').innerText = `Indexed Images: ${data.total_indexed}`;
            } catch (err) {
                status.innerText = "Failed to upload image.";
            }
        }

        // Auto-load initial search
        document.getElementById('query-input').value = 'a cute dog playing on grass';
        handleSearch(new Event('submit'));
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
