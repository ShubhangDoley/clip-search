import open_clip
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
    print(f"Running OpenCLIP ViT-B/32 on device: {device}")
    
    img_path = get_or_create_sample_image("test.jpg")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device).eval()

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    candidate_labels = ["a photo of a dog", "a blue graphic image with text", "a red sports car", "a landscape painting"]
    text = tokenizer(candidate_labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    print("\nCandidate Probabilities:")
    for label, prob in zip(candidate_labels, similarity[0].cpu().numpy()):
        print(f"  {label:35s}: {prob:.4f}")

if __name__ == "__main__":
    main()
