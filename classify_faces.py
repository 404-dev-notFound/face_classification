import os
import shutil
import sys
from pathlib import Path
import face_recognition # Standard library
import click

# Add the current directory to sys.path
sys.path.append(str(Path(__file__).parent))

def load_known_faces(known_dir):
    known_encodings = []
    known_names = []
    known_path = Path(known_dir)
    
    print(f"Loading known faces from {known_dir}...")
    
    for item in known_path.iterdir():
        if item.is_dir():
            name = item.name
            for img_file in item.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    try:
                        image = face_recognition.load_image_file(str(img_file))
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_encodings.append(encodings[0])
                            known_names.append(name)
                    except Exception as e:
                        print(f"Skipping {img_file}: {e}")
                        
        elif item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            name = item.stem
            try:
                image = face_recognition.load_image_file(str(item))
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
            except Exception as e:
                print(f"Skipping {item}: {e}")
                
    print(f"Loaded {len(known_encodings)} known face encodings.")
    return known_names, known_encodings

@click.command()
@click.option('--known-dir', required=True, help='Directory containing images of known people.')
@click.option('--unknown-dir', required=True, help='Directory containing images to classify.')
@click.option('--output-dir', default='output', help='Directory where classified images will be moved.')
@click.option('--tolerance', default=0.45, help='Strictness (0.45=Very Strict, 0.50=Strict, 0.60=Loose).')
@click.option('--use-cnn', is_flag=True, help='Use CNN-based face detection.')
def classify_and_sort(known_dir, unknown_dir, output_dir, tolerance, use_cnn):
    known_names, known_encodings = load_known_faces(known_dir)
    
    if not known_encodings:
        print("No known faces found. Please check your known-dir.")
        return

    unknown_path = Path(unknown_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for name in set(known_names):
        (output_path / name).mkdir(exist_ok=True)
    (output_path / "unknown").mkdir(exist_ok=True)

    print(f"Scanning images in {unknown_dir}...")
    model = "cnn" if use_cnn else "hog"
    
    for img_file in unknown_path.glob("*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
            
        print(f"Processing {img_file.name}...")
        
        try:
            image = face_recognition.load_image_file(str(img_file))
            height, width = image.shape[:2]

            # --- SMART DYNAMIC LOGIC ---
            # Goal: Maximize detection accuracy without crashing 8GB VRAM.
            
            # Logic: If image is huge (> 2 Megapixels), the faces are already big enough.
            # We do NOT need to upsample (which would crash VRAM).
            # If image is small (< 2 MP), we DO need to upsample to find details.
            
            pixel_count = height * width
            
            if model == "cnn" and pixel_count > 2_000_000: # Approx 1080p limit
                upsample_times = 0 
                # print(f"  > High-Res image ({width}x{height}). Mode: Safe (Upsample=0)")
            else:
                upsample_times = 1
                # print(f"  > Standard-Res image ({width}x{height}). Mode: Detail (Upsample=1)")
            
            # ---------------------------

            face_locs = face_recognition.face_locations(image, number_of_times_to_upsample=upsample_times, model=model)
            encodings = face_recognition.face_encodings(image, known_face_locations=face_locs)
            
            if not encodings:
                print(f"  - No face detected.")
                shutil.copy(str(img_file), output_path / "unknown" / img_file.name)
                continue

            found_match = False
            for encoding in encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                matches = distances <= tolerance
                
                if any(matches):
                    best_match_index = distances.argmin()
                    name = known_names[best_match_index]
                    dist_score = distances[best_match_index]
                    
                    shutil.copy(str(img_file), output_path / name / img_file.name)
                    found_match = True
                    print(f"  - Found {name} (Confidence: {dist_score:.3f})")
                    break 

            if not found_match:
                best_dist = distances.min() if len(distances) > 0 else 1.0
                print(f"  - Unknown Person (Best guess: {best_dist:.3f})")
                shutil.copy(str(img_file), output_path / "unknown" / img_file.name)

        except Exception as e:
            print(f"  - Error processing file: {e}")
            continue

    print(f"Task complete! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    classify_and_sort()