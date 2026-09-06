import os
import argparse
import numpy as np
import torch
import open_clip
from PIL import Image
import faiss
from tqdm import tqdm

def build_index(img_dir, output_index="image_index.faiss", output_paths="paths.npy", batch_size=32, model_name="ViT-B-32", pretrained="openai"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Building index for directory '{img_dir}' using {model_name} on {device}...")
    
    if not os.path.exists(img_dir):
        print(f"Error: Directory '{img_dir}' does not exist.")
        return False
        
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in os.listdir(img_dir) if os.path.splitext(f.lower())[1] in valid_exts]
    
    if not image_files:
        print(f"No valid images found in '{img_dir}'.")
        return False
        
    print(f"Found {len(image_files)} images to index.")
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    
    embeddings = []
    processed_paths = []
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_tensors = []
        batch_paths = []
        
        for fname in batch_files:
            fpath = os.path.join(img_dir, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                tensor = preprocess(img)
                batch_tensors.append(tensor)
                batch_paths.append(fname)
            except Exception as e:
                print(f"Skipping corrupted image {fname}: {e}")
                
        if not batch_tensors:
            continue
            
        input_tensors = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            embs = model.encode_image(input_tensors)
            embs /= embs.norm(dim=-1, keepdim=True)
            
        embeddings.append(embs.cpu().numpy())
        processed_paths.extend(batch_paths)
        
    if not embeddings:
        print("No embeddings were successfully computed.")
        return False
        
    embeddings_matrix = np.vstack(embeddings).astype('float32')
    dim = embeddings_matrix.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)
    
    faiss.write_index(index, output_index)
    np.save(output_paths, np.array(processed_paths))
    
    print(f"Successfully indexed {len(processed_paths)} images!")
    print(f"Saved FAISS index to '{output_index}' (Dim: {dim}, Total: {index.ntotal})")
    print(f"Saved path map to '{output_paths}'")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector index for an image directory.")
    parser.add_argument("--img_dir", type=str, default="test_images", help="Path to image directory")
    parser.add_argument("--output_index", type=str, default="image_index.faiss", help="Output FAISS index path")
    parser.add_argument("--output_paths", type=str, default="paths.npy", help="Output paths numpy file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    
    args = parser.parse_args()
    build_index(args.img_dir, args.output_index, args.output_paths, args.batch_size)
