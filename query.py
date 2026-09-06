import os
import argparse
import numpy as np
import torch
import open_clip
import faiss

def search_index(query_text, top_k=5, index_path="image_index.faiss", paths_path="paths.npy", model_name="ViT-B-32", pretrained="openai"):
    if not os.path.exists(index_path) or not os.path.exists(paths_path):
        print(f"Error: Index file '{index_path}' or paths file '{paths_path}' not found.")
        print("Please run 'build_index.py' first.")
        return []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    index = faiss.read_index(index_path)
    paths = np.load(paths_path)

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    text_tokens = tokenizer([query_text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        
    text_emb_np = text_emb.cpu().numpy().astype('float32')

    k = min(top_k, index.ntotal)
    scores, indices = index.search(text_emb_np, k)

    results = []
    print(f"\nQuery: '{query_text}' | Top-{k} Results:")
    print("-" * 65)
    print(f"{'Rank':<6} | {'Similarity Score':<18} | {'Image File'}")
    print("-" * 65)
    
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        file_path = paths[idx]
        print(f"{rank:<6} | {score:<18.4f} | {file_path}")
        results.append((file_path, float(score)))
        
    print("-" * 65)
    return results

def main():
    parser = argparse.ArgumentParser(description="Query semantic image search index.")
    parser.add_argument("--prompt", type=str, help="Text search query (e.g. 'a dog on a sofa')")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--index", type=str, default="image_index.faiss", help="FAISS index path")
    parser.add_argument("--paths", type=str, default="paths.npy", help="Paths numpy array file")
    
    args = parser.parse_args()

    if args.prompt:
        search_index(args.prompt, top_k=args.top_k, index_path=args.index, paths_path=args.paths)
    else:
        print("Semantic Image Search CLI (type 'exit' or 'quit' to stop)")
        while True:
            try:
                user_input = input("\nEnter search prompt > ").strip()
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    break
                search_index(user_input, top_k=args.top_k, index_path=args.index, paths_path=args.paths)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
