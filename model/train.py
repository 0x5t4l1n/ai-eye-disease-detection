#!/usr/bin/env python3
"""
Optimized Eye Disease Detection Training Script
- Fixed augmentation method accessibility
- Memory-efficient data handling
- Enhanced model architecture
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
import tensorflow_addons as tfa

# Check if required packages are available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    import seaborn as sns
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please run: pip install tensorflow tensorflow-addons opencv-python scikit-learn matplotlib tqdm seaborn")
    sys.exit(1)

print("✅ All required packages are available!")

class BalancedDataGenerator(keras.utils.Sequence):
    """Memory-efficient data generator with on-the-fly augmentation"""
    def __init__(self, X, y, class_weights, batch_size=32, augment=True):
        self.X = X
        self.y = y
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = []
        batch_y = []
        
        for i in batch_indices:
            img = self.X[i]
            if self.augment:
                img = self.random_augment(img.copy())
            batch_x.append(img)
            batch_y.append(self.y[i])
            
        sample_weights = np.array([self.class_weights[y] for y in batch_y])
        return np.array(batch_x), np.array(batch_y), sample_weights
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def random_augment(self, image):
        """Apply advanced random augmentations to an image"""
        # Convert to uint8 for OpenCV operations
        img = (image * 255).astype(np.uint8)
        
        # Geometric transformations
        if np.random.random() < 0.8:
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Color transformations
        if np.random.random() < 0.7:
            alpha = np.random.uniform(0.7, 1.5)  # Contrast
            beta = np.random.uniform(-30, 30)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random flips
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)  # Horizontal
        if np.random.random() < 0.3:
            img = cv2.flip(img, 0)  # Vertical
        
        # Noise and blur
        if np.random.random() < 0.4:
            noise = np.random.normal(0, np.random.uniform(0.5, 3.0), img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # Gamma correction
        if np.random.random() < 0.4:
            gamma = np.random.uniform(0.7, 1.5)
            table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)
        
        # Convert back to float32 and normalize
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return img

class EnhancedFundusTrainer:
    def __init__(self):
        self.dataset_path = "fundus_dataset_medical"
        self.model_path = "models"
        self.image_size = (224, 224)
        self.batch_size = 32 if len(tf.config.list_physical_devices('GPU')) > 0 else 16
        
        # Create directories
        Path(self.model_path).mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
    
    def enhanced_preprocess_image(self, image_path):
        """Enhanced image preprocessing with better techniques"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to RGB and apply CLAHE
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Resize and normalize
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            image = image.astype(np.float32) / 255.0
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            return image
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def random_augment(self, image):
        """Augmentation method accessible to both trainer and generator"""
        # Create temporary generator instance to use its augmentation
        temp_gen = BalancedDataGenerator([], [], {}, augment=True)
        return temp_gen.random_augment(image)
    
    def load_dataset_with_balancing(self):
        """Load and balance dataset with memory efficiency"""
        print("📁 Loading and balancing dataset...")
        
        dataset_path = Path(self.dataset_path)
        if (dataset_path / "1000images").exists():
            dataset_path = dataset_path / "1000images"

        # First pass: count images per class
        class_counts = {}
        for category_dir in dataset_path.iterdir():
            if category_dir.is_dir():
                class_counts[category_dir.name] = len(list(category_dir.glob('*.*')))
        
        print("\nInitial class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        
        # Calculate target samples (median-based)
        target_samples = int(np.median(list(class_counts.values()))) * 2
        min_samples = 100
        
        # Second pass: load and balance
        label_encoder = LabelEncoder()
        label_encoder.fit(list(class_counts.keys()))
        
        X, y = [], []
        
        for class_name in class_counts:
            current_count = class_counts[class_name]
            needed_samples = max(min_samples, min(target_samples, current_count * 2))
            
            print(f"Class {class_name}: {current_count} -> {needed_samples} samples")
            
            # Load original images
            class_images = []
            for img_path in (dataset_path / class_name).glob('*.*'):
                img = self.enhanced_preprocess_image(img_path)
                if img is not None:
                    class_images.append(img)
                    if len(class_images) >= needed_samples:
                        break
            
            # Add augmented samples if needed
            if len(class_images) < needed_samples:
                additional = needed_samples - len(class_images)
                augmented = [self.random_augment(class_images[np.random.randint(0, len(class_images))].copy()) 
                            for _ in range(additional)]
                class_images.extend(augmented)
            
            X.extend(class_images)
            y.extend([class_name] * len(class_images))
        
        y_encoded = label_encoder.transform(y)
        print(f"✅ Balanced dataset: {len(X)} images, {len(label_encoder.classes_)} categories")
        return np.array(X), y_encoded, label_encoder.classes_
    
    def create_enhanced_model(self, num_classes):
        """Create optimized model architecture"""
        print("🧠 Creating enhanced model...")
        
        # Base model with fine-tuning
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(*self.image_size, 3))
        
        # Data augmentation layers
        x = layers.RandomFlip("horizontal_and_vertical")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomContrast(0.2)(x)
        
        # Feature extraction
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.Dense(256, activation='swish')(x)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        x = layers.multiply([x, attention])
        
        # Classification head
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Optimizer with weight decay
        optimizer = tfa.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5
        )
        
        # Focal loss for class imbalance
        loss = tfa.losses.SigmoidFocalCrossEntropy(
            from_logits=False,
            alpha=0.25,
            gamma=2.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ],
            weighted_metrics=[
                tfa.metrics.F1Score(num_classes=num_classes, average='weighted')
            ]
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def train_with_enhanced_strategies(self, model, X_train, y_train, X_val, y_val, class_names, epochs=50):
        """Optimized training process"""
        print("🚀 Training with enhanced strategies...")
        
        # Calculate boosted class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight * 1.5 for i, weight in enumerate(class_weights)}
        
        print("Class weights:")
        for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
            print(f"  {class_name}: {weight:.3f} -> {weight*1.5:.3f}")
        
        # Create data generators
        train_gen = BalancedDataGenerator(
            X_train, y_train, class_weight_dict,
            batch_size=self.batch_size, augment=True
        )
        val_gen = BalancedDataGenerator(
            X_val, y_val, {i:1.0 for i in range(len(class_names))},
            batch_size=self.batch_size, augment=False
        )
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_f1_score',
                patience=15,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                f"{self.model_path}/best_model.h5",
                monitor='val_f1_score',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, class_names):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model...")
        
        # Get predictions and metrics
        predictions = model.evaluate(X_test, y_test, verbose=0)
        test_loss, test_acc, test_top3, test_f1 = predictions[:4]
        
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Top-3 Accuracy: {test_top3:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        report = classification_report(
            y_test, y_pred_classes, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        print(classification_report(y_test, y_pred_classes, target_names=class_names, zero_division=0))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes, class_names)
        
        return test_acc, test_loss, report
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Visualize confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 10}
        )
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Confusion matrix saved to plots/confusion_matrix.png")
    
    def plot_training_history(self, history):
        """Visualize training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ('accuracy', 'val_accuracy', 'Accuracy'),
            ('loss', 'val_loss', 'Loss'),
            ('top_3_accuracy', 'val_top_3_accuracy', 'Top-3 Accuracy'),
            ('f1_score', 'val_f1_score', 'F1 Score')
        ]
        
        for i, (train_metric, val_metric, title) in enumerate(metrics):
            ax = axes[i//2, i%2]
            ax.plot(history.history[train_metric], label=f'Training {title}')
            ax.plot(history.history[val_metric], label=f'Validation {title}')
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Training history saved to plots/training_history.png")
    
    def save_model(self, model, class_names, accuracy, report):
        """Save model and metadata"""
        print("💾 Saving model...")
        
        # Save model
        model_file = f"{self.model_path}/eye_disease_model.keras"
        model.save(model_file)
        
        # Save metadata
        metadata = {
            'model_file': model_file,
            'num_classes': len(class_names),
            'image_size': self.image_size,
            'test_accuracy': float(accuracy),
            'class_names': class_names.tolist(),
            'classification_report': report,
            'model_architecture': 'DenseNet121 + Enhanced Head',
            'training_features': [
                'Memory-efficient balancing',
                'On-the-fly augmentation',
                'Focal loss',
                'Attention mechanism',
                'AdamW optimizer'
            ]
        }
        
        with open(f"{self.model_path}/model_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(f"{self.model_path}/classes.json", 'w') as f:
            json.dump(class_names.tolist(), f, indent=2)
        
        print(f"✅ Model saved to: {model_file}")
    
    def run_training(self):
        """Complete training pipeline"""
        print("🎯 Eye Disease Detection Training Pipeline")
        print("=" * 60)
        
        # Load and balance dataset
        X, y, class_names = self.load_dataset_with_balancing()
        if X is None:
            return False
        
        # Train/val/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\nDataset splits:")
        print(f"Train: {len(X_train)}")
        print(f"Validation: {len(X_val)}")
        print(f"Test: {len(X_test)}")
        
        # Create and train model
        model = self.create_enhanced_model(len(class_names))
        history = self.train_with_enhanced_strategies(
            model, X_train, y_train, X_val, y_val, class_names
        )
        
        # Evaluate and save
        accuracy, loss, report = self.evaluate_model(model, X_test, y_test, class_names)
        self.save_model(model, class_names, accuracy, report)
        self.plot_training_history(history)
        
        print("\n🎉 Training completed successfully!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        return True

def main():
    """Main execution function"""
    print("Optimized Eye Disease Detection Training")
    print("=" * 50)
    
    # System info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Addons version: {tfa.__version__}")
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    print(f"GPU available: {gpu_count} {'(✅)' if gpu_count > 0 else '(❌ CPU only)'}")
    
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    # Run training
    trainer = EnhancedFundusTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("\nOutput files:")
        print("- models/eye_disease_model.keras")
        print("- models/model_info.json")
        print("- models/classes.json")
        print("- plots/confusion_matrix.png")
        print("- plots/training_history.png")
    else:
        print("\n❌ Training failed")

if __name__ == "__main__":
    main()