from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image, ImageDraw
import os

def get_or_create_sample_image(path="test.jpg"):
    if not os.path.exists(path):
        img = Image.new('RGB', (300, 300), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Sample Test Image", fill=(255, 255, 0))
        img.save(path)
        print(f"Created sample test image at {path}")
    return path

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running SigLIP base on device: {device}")
    
    img_path = get_or_create_sample_image("test.jpg")
    
    model_id = "google/siglip-base-patch16-224"
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    image = Image.open(img_path)
    candidate_labels = ["a photo of a dog", "a blue graphic image with text", "a red sports car", "a landscape painting"]

    inputs = processor(text=candidate_labels, images=image, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits_per_image)

    print("\nCandidate Probabilities:")
    for label, prob in zip(candidate_labels, probs[0].cpu().numpy()):
        print(f"  {label:35s}: {prob:.4f}")

if __name__ == "__main__":
    main()
