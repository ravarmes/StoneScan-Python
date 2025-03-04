import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import os
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def load_and_prepare_data():
    """
    Carrega e prepara os dados de treinamento.
    Por enquanto, usa dados sintéticos, mas deve ser substituído por dados reais.
    """
    logging.info("Loading and preparing data...")
    
    # TODO: Substituir por carregamento de dados reais
    def generate_synthetic_data(num_samples=1000, img_size=(224, 224, 3), num_classes=2):
        X = np.random.rand(num_samples, *img_size).astype(np.float32)
        y = np.random.randint(0, num_classes, num_samples)
        return X, tf.keras.utils.to_categorical(y, num_classes)
    
    # Gerando dados sintéticos (temporário)
    X_train, y_train = generate_synthetic_data(800)
    X_test, y_test = generate_synthetic_data(200)
    
    # Normalização
    normalization_layer = layers.Rescaling(1./255)
    X_train = normalization_layer(X_train)
    X_test = normalization_layer(X_test)
    
    return X_train, y_train, X_test, y_test

def create_data_augmentation():
    """Cria pipeline de data augmentation."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

def create_callbacks():
    """Cria callbacks para monitoramento e controle do treinamento."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]

def create_model():
    """
    Cria o modelo baseado no MobileNetV2 pré-treinado.
    A base é inicialmente congelada para treinamento das camadas superiores.
    """
    logging.info("Creating model with pre-trained MobileNetV2 base...")
    
    # Carregar modelo base pré-treinado
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'  # Carregar pesos pré-treinados do ImageNet
    )
    base_model.trainable = False  # Congelar a base inicialmente
    
    # Criar modelo completo
    model = models.Sequential([
        create_data_augmentation(),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    
    return model, base_model

def train_top_layers(model, X_train, y_train, X_test, y_test, callbacks):
    """
    Primeira fase: treinar apenas as camadas superiores.
    A base do modelo permanece congelada.
    """
    logging.info("Starting initial training phase (top layers only)...")
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        batch_size=32
    )
    
    return history

def fine_tune_model(model, base_model, X_train, y_train, X_test, y_test, callbacks):
    """
    Segunda fase: fine-tuning das últimas camadas da base.
    Descongela e treina as últimas camadas do modelo base.
    """
    logging.info("Starting fine-tuning phase...")
    
    # Descongelar as últimas camadas da base
    tuning_layers = len(base_model.layers) // 2
    for layer in base_model.layers[-tuning_layers:]:
        layer.trainable = True
    
    # Recompilar com learning rate menor para fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar novamente
    history_fine = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        batch_size=32
    )
    
    return history_fine

def plot_training_history(history, history_fine=None):
    """Plota os gráficos de treinamento."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Initial Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Initial Validation Accuracy')
    if history_fine:
        plt.plot(range(len(history.history['accuracy']), 
                      len(history.history['accuracy']) + len(history_fine.history['accuracy'])),
                history_fine.history['accuracy'], label='Fine-tuning Training Accuracy')
        plt.plot(range(len(history.history['val_accuracy']), 
                      len(history.history['val_accuracy']) + len(history_fine.history['val_accuracy'])),
                history_fine.history['val_accuracy'], label='Fine-tuning Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Initial Training Loss')
    plt.plot(history.history['val_loss'], label='Initial Validation Loss')
    if history_fine:
        plt.plot(range(len(history.history['loss']), 
                      len(history.history['loss']) + len(history_fine.history['loss'])),
                history_fine.history['loss'], label='Fine-tuning Training Loss')
        plt.plot(range(len(history.history['val_loss']), 
                      len(history.history['val_loss']) + len(history_fine.history['val_loss'])),
                history_fine.history['val_loss'], label='Fine-tuning Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    try:
        # Criar diretórios necessários
        os.makedirs('logs', exist_ok=True)
        
        # Carregar e preparar dados
        X_train, y_train, X_test, y_test = load_and_prepare_data()
        
        # Criar modelo e callbacks
        model, base_model = create_model()
        callbacks = create_callbacks()
        
        # Primeira fase: treinar apenas as camadas superiores
        history = train_top_layers(model, X_train, y_train, X_test, y_test, callbacks)
        
        # Segunda fase: fine-tuning
        history_fine = fine_tune_model(model, base_model, X_train, y_train, X_test, y_test, callbacks)
        
        # Salvar modelo
        logging.info("Saving final model...")
        model.save('stone_scan_model.keras')
        
        # Avaliação final
        logging.info("Evaluating model...")
        loss, acc = model.evaluate(X_test, y_test)
        logging.info(f"Final Test Accuracy: {acc:.4f}")
        
        # Plotar resultados
        logging.info("Plotting training history...")
        plot_training_history(history, history_fine)
        
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 