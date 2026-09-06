import os
from PIL import Image, ImageDraw, ImageFont

def generate_sample_dataset(output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        ("dog.jpg", (255, 200, 150), "A cute golden dog sitting on green grass", (34, 139, 34)),
        ("red_sports_car.jpg", (220, 20, 60), "A fast red sports car on a highway", (50, 50, 50)),
        ("sunset_beach.jpg", (255, 140, 0), "A beautiful ocean sunset on a sandy beach", (30, 144, 255)),
        ("cat.jpg", (200, 200, 200), "A fluffy cat resting near a cozy window", (139, 69, 19)),
        ("person_bicycle.jpg", (100, 149, 237), "A person riding a bicycle in a city park", (46, 139, 87))
    ]
    
    for filename, bg_color, text_desc, banner_color in samples:
        filepath = os.path.join(output_dir, filename)
        img = Image.new('RGB', (400, 400), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw decorative shapes
        draw.rectangle([20, 20, 380, 100], fill=banner_color)
        draw.ellipse([100, 150, 300, 350], fill=(255, 255, 255))
        draw.text((30, 40), text_desc, fill=(255, 255, 255))
        
        img.save(filepath)
        print(f"Generated sample image: {filepath}")

    print(f"\nSample dataset ready in directory '{output_dir}' ({len(samples)} images).")

if __name__ == "__main__":
    generate_sample_dataset()
