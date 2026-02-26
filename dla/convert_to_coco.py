import json
import os
import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import imagesize

def main():
    parser = argparse.ArgumentParser(description="Convert parser outputs to COCO format")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory of parser outputs (e.g., outputs_dla/2023_math_odd)")
    parser.add_argument("--output_path", type=str, default="coco_dataset/annotations/instances_default.json", help="Path to save COCO JSON")
    parser.add_argument("--image_root_rel", type=str, default="", help="Relative path from json output to image root if needed. If empty, uses absolute paths or assumes images are co-located.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # COCO Structure
    coco = {
        "info": {
            "description": "DLA Dataset for Math Problems",
            "version": "1.0",
            "year": 2024,
            "contributor": "Antigravity",
            "date_created": "2024-02-09"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "problem", "supercategory": "layout"}
        ]
    }

    # Find all json files in the input directory (recursive)
    # Parser structure: outputs_dla/group_name/page_id/page_id.json
    json_files = list(input_dir.glob("**/*.json"))
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    annotation_id = 1
    image_id_map = {} # path -> id
    next_image_id = 1

    for json_file in tqdm(json_files, desc="Converting to COCO"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if this is a valid page payload
            if "page_image" not in data or "questions" not in data:
                continue

            image_path = data["page_image"]
            # 파일명만 추출 (경로에 _gpu_chunk_X 등 임시 디렉토리가 포함될 수 있으므로)
            file_name = Path(image_path).name
            
            # Read dimensions if not present (backward compatibility)
            width = data.get("width")
            height = data.get("height")
            
            if width is None or height is None:
                # Fallback: try to read from file
                try:
                    width, height = imagesize.get(image_path)
                except:
                    # If image path is not valid/found, skip or assume defaults?
                    # Let's try to assume relative to json if absolute fails
                    try:
                        # parser saves image path, usually absolute or relative to run dir
                        # We try to resolve it
                        if os.path.exists(image_path):
                            width, height = imagesize.get(image_path)
                        else:
                            # Try finding it in the raw folder sibling to json?
                            # parser output: .../page_id.json
                            # parser input: .../page_id.png (maybe in data dir?)
                            print(f"Warning: Could not determine dimensions for {image_path}, skipping.")
                            continue
                    except:
                        continue

            if file_name not in image_id_map:
                image_id = next_image_id
                next_image_id += 1
                image_id_map[file_name] = image_id
                
                coco["images"].append({
                    "id": image_id,
                    "file_name": file_name,  # 파일명만 저장 (image_dir과 결합하여 사용)
                    "width": width,
                    "height": height
                })
            else:
                image_id = image_id_map[file_name]

            for q in data["questions"]:
                # bbox is [x1, y1, x2, y2]
                bbox = q["bbox"]
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                
                # COCO bbox format is [x, y, width, height]
                coco_bbox = [x1, y1, w, h]
                area = w * h

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1, # problem
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [] # No segmentation
                })
                annotation_id += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    
    print(f"Saved COCO JSON to {output_path}")
    print(f"Total images: {len(coco['images'])}")
    print(f"Total annotations: {len(coco['annotations'])}")

if __name__ == "__main__":
    main()
