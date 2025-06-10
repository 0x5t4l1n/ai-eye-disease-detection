#!/usr/bin/env python3
"""
Dataset Reorganization Script
Fixes the nested directory structure and class imbalance
"""

import os
import shutil
from pathlib import Path
from collections import Counter
import json

def reorganize_dataset():
    """Reorganize the dataset to fix the nested structure"""
    
    print("🔧 DATASET REORGANIZATION")
    print("=" * 40)
    
    source_dir = Path("fundus_dataset/1000images")
    target_dir = Path("fundus_dataset_fixed")
    
    if not source_dir.exists():
        print("❌ Source directory doesn't exist!")
        return False
    
    # Create target directory
    target_dir.mkdir(exist_ok=True)
    
    # Get all subdirectories (disease categories)
    categories = [d for d in source_dir.iterdir() if d.is_dir() and d.name != "1000images"]
    
    print(f"Found {len(categories)} disease categories")
    
    category_counts = {}
    total_images = 0
    
    # Process each category
    for category in categories:
        category_name = category.name
        
        # Skip the problematic nested folder
        if category_name == "1000images":
            continue
            
        print(f"Processing: {category_name}")
        
        # Create target category directory
        target_category = target_dir / category_name
        target_category.mkdir(exist_ok=True)
        
        # Find all images in this category
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend(category.glob(f"*{ext}"))
            image_files.extend(category.glob(f"*{ext.upper()}"))
        
        # Copy images
        copied_count = 0
        for img_file in image_files:
            try:
                target_file = target_category / img_file.name
                shutil.copy2(img_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {img_file}: {e}")
        
        category_counts[category_name] = copied_count
        total_images += copied_count
        print(f"  Copied {copied_count} images")
    
    print(f"\n📊 REORGANIZATION SUMMARY:")
    print(f"Total images processed: {total_images}")
    print(f"Total categories: {len(category_counts)}")
    
    # Show category distribution
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} images")
    
    return True, category_counts

def filter_categories_by_size(min_images=10):
    """Filter categories to only include those with enough samples"""
    
    print(f"\n🔍 FILTERING CATEGORIES (minimum {min_images} images)")
    print("=" * 50)
    
    source_dir = Path("fundus_dataset_fixed")
    target_dir = Path("fundus_dataset_balanced")
    
    if not source_dir.exists():
        print("❌ Fixed dataset doesn't exist!")
        return False
    
    target_dir.mkdir(exist_ok=True)
    
    valid_categories = []
    
    for category_dir in source_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        # Count images
        image_count = len(list(category_dir.glob("*.jpg")) + 
                         list(category_dir.glob("*.jpeg")) + 
                         list(category_dir.glob("*.png")))
        
        if image_count >= min_images:
            print(f"✅ {category_dir.name}: {image_count} images - KEEPING")
            
            # Copy entire category
            target_category = target_dir / category_dir.name
            if target_category.exists():
                shutil.rmtree(target_category)
            shutil.copytree(category_dir, target_category)
            valid_categories.append((category_dir.name, image_count))
        else:
            print(f"❌ {category_dir.name}: {image_count} images - REMOVING (too few)")
    
    print(f"\n📊 FINAL DATASET:")
    print(f"Valid categories: {len(valid_categories)}")
    total_images = sum(count for _, count in valid_categories)
    print(f"Total images: {total_images}")
    
    return True, valid_categories

def create_simplified_categories():
    """Create simplified medical categories"""
    
    print(f"\n🏥 CREATING SIMPLIFIED MEDICAL CATEGORIES")
    print("=" * 45)
    
    source_dir = Path("fundus_dataset_fixed")
    target_dir = Path("fundus_dataset_medical")
    
    if not source_dir.exists():
        print("❌ Fixed dataset doesn't exist!")
        return False
    
    target_dir.mkdir(exist_ok=True)
    
    # Define medical category mappings
    medical_categories = {
        'normal': ['0.0.Normal', '0.1.Tessellated fundus'],
        'diabetic_retinopathy': ['0.3.DR1', '1.0.DR2', '1.1.DR3'],
        'glaucoma': ['10.0.Possible glaucoma', '10.1.Optic atrophy', '0.2.Large optic cup'],
        'vascular_disorders': ['2.0.BRVO', '2.1.CRVO', '11.Severe hypertensive retinopathy'],
        'macular_disorders': ['6.Maculopathy', '7.ERM', '8.MH', '5.0.CSCR'],
        'retinal_detachment': ['4.Rhegmatogenous RD'],
        'other_pathology': []  # Will collect remaining categories
    }
    
    # Create target directories
    for med_cat in medical_categories.keys():
        (target_dir / med_cat).mkdir(exist_ok=True)
    
    category_counts = {}
    processed_categories = set()
    
    # Map existing categories to medical categories
    for med_cat, orig_categories in medical_categories.items():
        count = 0
        for orig_cat in orig_categories:
            source_cat_dir = source_dir / orig_cat
            if source_cat_dir.exists():
                processed_categories.add(orig_cat)
                
                # Copy all images from original category
                target_cat_dir = target_dir / med_cat
                
                for img_file in source_cat_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        new_name = f"{orig_cat}_{img_file.name}"
                        shutil.copy2(img_file, target_cat_dir / new_name)
                        count += 1
        
        category_counts[med_cat] = count
        print(f"{med_cat}: {count} images")
    
    # Handle remaining categories
    other_count = 0
    for category_dir in source_dir.iterdir():
        if category_dir.is_dir() and category_dir.name not in processed_categories:
            target_cat_dir = target_dir / 'other_pathology'
            
            for img_file in category_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    new_name = f"{category_dir.name}_{img_file.name}"
                    shutil.copy2(img_file, target_cat_dir / new_name)
                    other_count += 1
    
    category_counts['other_pathology'] = other_count
    print(f"other_pathology: {other_count} images")
    
    # Remove empty categories
    final_categories = {k: v for k, v in category_counts.items() if v > 0}
    
    for cat_name, count in final_categories.items():
        if count == 0:
            shutil.rmtree(target_dir / cat_name)
    
    print(f"\n📊 MEDICAL CATEGORIES SUMMARY:")
    for cat, count in final_categories.items():
        if count > 0:
            print(f"  {cat}: {count} images")
    
    return True, final_categories

def main():
    """Main reorganization process"""
    
    print("🏥 EYE DISEASE DATASET REORGANIZER")
    print("=" * 50)
    
    # Step 1: Fix nested structure
    print("\n1️⃣ FIXING NESTED DIRECTORY STRUCTURE")
    success, _ = reorganize_dataset()
    if not success:
        return
    
    # Step 2: Create simplified medical categories
    print("\n2️⃣ CREATING MEDICAL CATEGORIES")
    success, medical_cats = create_simplified_categories()
    if not success:
        return
    
    # Step 3: Update training script path
    print("\n3️⃣ NEXT STEPS:")
    print("✅ Dataset reorganized successfully!")
    print("✅ Update your training script to use: 'fundus_dataset_medical'")
    print("✅ Expected categories:", list(medical_cats.keys()))
    
    # Save configuration
    config = {
        'dataset_path': 'fundus_dataset_medical',
        'categories': medical_cats,
        'total_images': sum(medical_cats.values()),
        'num_classes': len([k for k, v in medical_cats.items() if v > 0])
    }
    
    with open('dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to dataset_config.json")
    print(f"\n🎯 Ready for proper medical AI training!")

if __name__ == "__main__":
    main()
