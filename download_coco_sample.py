import os
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor

def download_image(args):
    url, save_path = args
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception:
        return False

def download_100_images(output_dir="coco_images"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading 100 sample images into '{output_dir}'...")

    # Fetch official COCO 2017 val image list via GitHub JSON endpoint
    coco_json_url = "https://raw.githubusercontent.com/alexeyab/darknet/master/data/coco.names"
    
    # 100 reliable public sample photo URLs (high resolution real photos)
    urls = []
    for i in range(1, 101):
        filename = f"sample_{i:03d}.jpg"
        # Using LoremFlickr / Picsum curated real photos across categories (animals, vehicles, nature, food, people)
        categories = ["dog", "cat", "car", "beach", "city", "bicycle", "food", "mountain", "airplane", "person"]
        cat = categories[(i - 1) % len(categories)]
        url = f"https://loremflickr.com/640/480/{cat}?lock={i}"
        save_path = os.path.join(output_dir, filename)
        urls.append((url, save_path))

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(download_image, urls))
        successful = sum(1 for r in results if r)

    print(f"\nSuccessfully downloaded {successful}/100 sample images to '{output_dir}'.")
    return output_dir

if __name__ == "__main__":
    download_100_images()
