#!/usr/bin/env python3
"""
Improved Eye Disease Detection Training Script
Addresses class imbalance, adds better data augmentation, and improves model architecture
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# Check if required packages are available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    from kaggle.api.kaggle_api_extended import KaggleApi
    import seaborn as sns
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please run: pip install tensorflow opencv-python scikit-learn kaggle matplotlib tqdm seaborn")
    sys.exit(1)

print("✅ All required packages are available!")

class ImprovedFundusTrainer:
    def __init__(self):
        self.dataset_path = "fundus_dataset_medical"
        self.model_path = "models"
        self.image_size = (224, 224)
        self.batch_size = 16  # Reduced for better gradient updates
        
        # Create directories
        Path(self.model_path).mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
    
    def download_dataset(self):
        """Download the Kaggle dataset"""
        print("📥 Downloading dataset from Kaggle...")
        
        try:
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                "linchundan/fundusimage1000",
                path=self.dataset_path,
                unzip=True
            )
            print("✅ Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("Please check your Kaggle API setup:")
            print("1. Ensure kaggle.json is in ~/.kaggle/")
            print("2. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def enhanced_preprocess_image(self, image_path):
        """Enhanced image preprocessing with better techniques"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Resize with anti-aliasing
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset_with_balancing(self):
        """Load dataset with improved balancing strategies"""
        print("📁 Loading dataset with class balancing...")
        
        images = []
        labels = []
        
        dataset_path = Path(self.dataset_path)
        
        # Check for nested directory
        if (dataset_path / "1000images").exists():
            dataset_path = dataset_path / "1000images"
        
        # Load images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        class_counts = {}
        
        for category_dir in tqdm(subdirs, desc="Loading categories"):
            category_name = category_dir.name
            
            # Get image files
            image_files = []
            for ext in image_extensions:
                image_files.extend(category_dir.rglob(f"*{ext}"))
                image_files.extend(category_dir.rglob(f"*{ext.upper()}"))
            
            class_images = []
            for image_file in image_files:
                processed_image = self.enhanced_preprocess_image(image_file)
                if processed_image is not None:
                    class_images.append(processed_image)
            
            if class_images:
                images.extend(class_images)
                labels.extend([category_name] * len(class_images))
                class_counts[category_name] = len(class_images)
        
        if len(images) == 0:
            return None, None, None
        
        # Print initial class distribution
        print("\nInitial class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        
        # Apply data balancing strategies
        X, y = self.balance_dataset(images, labels, class_counts)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"✅ Balanced dataset: {len(X)} images, {len(label_encoder.classes_)} categories")
        
        return np.array(X), y_encoded, label_encoder.classes_
    
    def balance_dataset(self, images, labels, class_counts):
        """Apply multiple strategies to balance the dataset"""
        print("⚖️ Balancing dataset...")
        
        # Strategy 1: Oversample minority classes using augmentation
        target_samples = max(class_counts.values())  # Target the largest class size
        min_samples = 50  # Minimum samples per class after balancing
        
        # If target is too high, cap it
        if target_samples > 200:
            target_samples = 200
        
        balanced_images = []
        balanced_labels = []
        
        # Group images by class
        class_images = {}
        for img, label in zip(images, labels):
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(img)
        
        # Balance each class
        for class_name, class_imgs in class_images.items():
            current_count = len(class_imgs)
            needed_samples = max(min_samples, min(target_samples, current_count * 3))
            
            print(f"Class {class_name}: {current_count} -> {needed_samples} samples")
            
            # Add original images
            balanced_images.extend(class_imgs)
            balanced_labels.extend([class_name] * len(class_imgs))
            
            # Generate additional samples if needed
            if needed_samples > current_count:
                additional_needed = needed_samples - current_count
                augmented_images = self.generate_augmented_samples(class_imgs, additional_needed)
                balanced_images.extend(augmented_images)
                balanced_labels.extend([class_name] * len(augmented_images))
        
        print(f"Final balanced dataset: {len(balanced_images)} images")
        return balanced_images, balanced_labels
    
    def generate_augmented_samples(self, images, count_needed):
        """Generate augmented samples for minority classes"""
        augmented = []
        
        for i in range(count_needed):
            # Randomly select an image to augment
            base_img = images[i % len(images)].copy()
            
            # Apply random augmentations
            augmented_img = self.random_augment(base_img)
            augmented.append(augmented_img)
        
        return augmented
    
    def random_augment(self, image):
        """Apply random augmentations to an image"""
        # Convert to uint8 for OpenCV operations
        img = (image * 255).astype(np.uint8)
        
        # Random rotation
        if np.random.random() < 0.7:
            angle = np.random.uniform(-20, 20)
            center = (img.shape[1]//2, img.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Random brightness/contrast
        if np.random.random() < 0.6:
            alpha = np.random.uniform(0.8, 1.3)  # Contrast
            beta = np.random.uniform(-20, 20)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random horizontal flip
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
        
        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # Convert back to float32 and normalize
        return img.astype(np.float32) / 255.0
    
    def create_improved_model(self, num_classes):
        """Create an improved model architecture"""
        print("🧠 Creating improved model...")
        
        # Use EfficientNetB0 as backbone
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Unfreeze top layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Create model with improved architecture
        inputs = keras.Input(shape=(*self.image_size, 3))
        
        # Data augmentation layer
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Use different learning rates for base and head
        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print(f"✅ Improved model created with {model.count_params():,} parameters")
        return model
    
    def train_with_class_weights(self, model, X_train, y_train, X_val, y_val, class_names, epochs=30):
        """Train model with class weights and improved callbacks"""
        print("🚀 Training with class weights...")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print("Class weights:")
        for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
            print(f"  {class_name}: {weight:.3f}")
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=6,
                factor=0.3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f"{self.model_path}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Advanced data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            brightness_range=[0.7, 1.3],
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        # Train with class weights
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return history
    
    def evaluate_with_detailed_metrics(self, model, X_test, y_test, class_names):
        """Comprehensive model evaluation"""
        print("📊 Detailed model evaluation...")
        
        # Get predictions
        predictions = model.evaluate(X_test, y_test, verbose=0)
        test_loss = predictions[0]
        test_accuracy = predictions[1]
        test_top3 = predictions[2] if len(predictions) > 2 else None
        
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        if test_top3 is not None:
            print(f"Test Top-3 Accuracy: {test_top3:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_test, y_pred_classes, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        print(classification_report(y_test, y_pred_classes, target_names=class_names, zero_division=0))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes, class_names)
        
        return test_accuracy, test_loss, report
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confusion matrix saved to plots/confusion_matrix.png")
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Loss
        axes[0,1].plot(history.history['loss'], label='Training Loss')
        axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Top-3 Accuracy
        if 'top_3_accuracy' in history.history:
            axes[1,0].plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
            axes[1,0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
            axes[1,0].set_title('Top-3 Accuracy')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Top-3 Accuracy')
            axes[1,0].legend()
            axes[1,0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1,1].plot(history.history['lr'])
            axes[1,1].set_title('Learning Rate')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Learning Rate')
            axes[1,1].set_yscale('log')
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training history saved to plots/training_history.png")
    
    def save_improved_model(self, model, class_names, accuracy, report):
        """Save model with comprehensive metadata"""
        print("💾 Saving improved model...")
        
        # Save model in Keras format
        model_file = f"{self.model_path}/eye_disease_model_improved.keras"
        model.save(model_file)
        
        # Save detailed metadata
        metadata = {
            'model_file': model_file,
            'num_classes': len(class_names),
            'image_size': self.image_size,
            'test_accuracy': float(accuracy),
            'class_names': class_names.tolist(),
            'classification_report': report,
            'model_architecture': 'EfficientNetB0 + Custom Head',
            'training_features': [
                'Class balancing with oversampling',
                'Advanced data augmentation',
                'Class weights',
                'Fine-tuning',
                'Enhanced preprocessing with CLAHE'
            ]
        }
        
        metadata_file = f"{self.model_path}/model_info_improved.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save class names
        classes_file = f"{self.model_path}/classes_improved.json"
        with open(classes_file, 'w') as f:
            json.dump(class_names.tolist(), f, indent=2)
        
        print(f"✅ Improved model saved to: {model_file}")
        print(f"✅ Metadata saved to: {metadata_file}")
        print(f"✅ Classes saved to: {classes_file}")
    
    def run_improved_training(self):
        """Run the complete improved training pipeline"""
        print("🎯 Improved Eye Disease Detection - Training Pipeline")
        print("=" * 60)
        
        # Step 1: Download dataset if needed
        if not os.path.exists(self.dataset_path):
            if not self.download_dataset():
                return False
        
        # Step 2: Load and balance dataset
        result = self.load_dataset_with_balancing()
        if result[0] is None:
            print("❌ Failed to load dataset. Exiting.")
            return False
        
        X, y, class_names = result
        
        if len(X) < 50:
            print(f"❌ Not enough images ({len(X)}). Need at least 50 for reliable training.")
            return False
        
        # Step 3: Stratified split
        print("✂️ Creating stratified splits...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Step 4: Create and train improved model
        model = self.create_improved_model(len(class_names))
        history = self.train_with_class_weights(
            model, X_train, y_train, X_val, y_val, class_names
        )
        
        # Step 5: Comprehensive evaluation
        accuracy, loss, report = self.evaluate_with_detailed_metrics(
            model, X_test, y_test, class_names
        )
        
        # Step 6: Save results
        self.save_improved_model(model, class_names, accuracy, report)
        self.plot_training_history(history)
        
        print("\n🎉 Improved training completed!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"Model and plots saved in respective directories")
        
        # Performance recommendations
        self.provide_performance_recommendations(accuracy, report)
        
        return True
    
    def provide_performance_recommendations(self, accuracy, report):
        """Provide recommendations based on performance"""
        print("\n📋 Performance Analysis & Recommendations:")
        print("-" * 50)
        
        if accuracy < 0.6:
            print("⚠️  Low accuracy detected. Consider:")
            print("   • Collecting more high-quality data")
            print("   • Trying different architectures (ResNet, DenseNet)")
            print("   • Adjusting hyperparameters")
        elif accuracy < 0.8:
            print("📈 Moderate performance. Improvements possible:")
            print("   • Fine-tune hyperparameters")
            print("   • Try ensemble methods")
            print("   • Experiment with different augmentation strategies")
        else:
            print("🎯 Good performance! Consider:")
            print("   • Cross-validation for robustness")
            print("   • Deployment optimization")
            print("   • Clinical validation")
        
        # Check for class-specific issues
        worst_classes = []
        for class_name in report:
            if isinstance(report[class_name], dict) and 'f1-score' in report[class_name]:
                if report[class_name]['f1-score'] < 0.5:
                    worst_classes.append(class_name)
        
        if worst_classes:
            print(f"\n🔍 Classes needing attention: {', '.join(worst_classes)}")
            print("   • Collect more samples for these classes")
            print("   • Review data quality for these categories")

def main():
    """Main function with improved error handling"""
    print("Improved Eye Disease Detection Training")
    print("Using enhanced techniques for better performance")
    print()
    
    # System info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    print()
    
    # Run improved training
    trainer = ImprovedFundusTrainer()
    success = trainer.run_improved_training()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("\nFiles created:")
        print("• models/eye_disease_model_improved.keras - Main model")
        print("• models/model_info_improved.json - Detailed metadata")
        print("• plots/confusion_matrix.png - Performance visualization")
        print("• plots/training_history.png - Training progress")
        print("\nNext steps:")
        print("1. Test model with validation images")
        print("2. Consider clinical validation")
        print("3. Deploy for production use")
    else:
        print("\n❌ Training failed. Check errors above.")

if __name__ == "__main__":
    main()