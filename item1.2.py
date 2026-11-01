import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================
RUTA_DATASET = "/home/chris/Downloads/PDI - MiniProyecto 2/Dataset Oficial"  # <-- Cambia esto por la ruta de tu carpeta principal
IMG_SIZE = (128, 128)            # Tamaño común para estandarizar (Requisito 1.2 y 1.3)
RANDOM_SEED = 42                 # Semilla para partición reproducible (Requisito 1.4)

# ==============================================================================
# ÍTEM 1.2: EXTRACCIÓN DE CARACTERÍSTICAS Y VISUALIZACIÓN
# ==============================================================================
def calcular_asimetria(canal):
    """Calcula la asimetría (skewness) estadística de un canal."""
    media = np.mean(canal)
    std = np.std(canal) + 1e-8
    return np.mean(((canal - media) / std) ** 3)

def extraer_descriptores(img_rgb):
    """
    Extrae el vector de 17 características de una imagen RGB preprocesada:
    - 9 de Color (Momentos HSV)
    - 5 de Textura (GLCM)
    - 3 de Bordes (Sobel + Canny)
    """
    # 1. Momentos de Color en HSV (9 valores)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(img_hsv)
    feats_color = []
    for canal in [h, s, v]:
        feats_color.extend([np.mean(canal), np.std(canal), calcular_asimetria(canal)])
        
    # Conversión a Escala de Grises para Textura y Bordes
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Textura mediante GLCM (5 valores)
    # Se calcula en 4 ángulos (0, 45, 90, 135 grados) a distancia 1 y se promedia
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    propiedades = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feats_textura = [np.mean(graycoprops(glcm, prop)) for prop in propiedades]
    
    # 3. Bordes mediante Sobel y Canny (3 valores)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitud_sobel = np.sqrt(sobelx**2 + sobely**2)
    
    bordes_canny = cv2.Canny(img_gray, 50, 150)
    densidad_canny = np.sum(bordes_canny > 0) / float(img_gray.size)
    
    feats_bordes = [np.mean(magnitud_sobel), np.std(magnitud_sobel), densidad_canny]
    
    # Combinar en un vector único de 17 dimensiones
    vector_17d = np.array(feats_color + feats_textura + feats_bordes, dtype=np.float32)
    return vector_17d, img_hsv, magnitud_sobel, bordes_canny

def visualizar_ejemplo_descriptores(img_rgb, nombre_clase):
    """Muestra las transformaciones visuales del Ítem 1.2 e imprime el vector de ejemplo."""
    vector, img_hsv, mag_sobel, canny = extraer_descriptores(img_rgb)
    
    nombres_feats = [
        "Media H", "Std H", "Skew H", "Media S", "Std S", "Skew S", "Media V", "Std V", "Skew V",
        "GLCM Contraste", "GLCM Disimilitud", "GLCM Homogeneidad", "GLCM Energía", "GLCM Correlación",
        "Media Sobel", "Std Sobel", "Densidad Canny"
    ]
    
    print(f"\n--- EJEMPLO DE VECTOR DE CARACTERÍSTICAS (Clase: {nombre_clase}) ---")
    for idx, (nombre, val) in enumerate(zip(nombres_feats, vector)):
        print(f"[{idx:02d}] {nombre:<20}: {val:.4f}")
        
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Ítem 1.2: Visualización de Características Extraídas ({nombre_clase})', fontsize=14)
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('ROI Preprocesada (RGB)')
    axes[1].imshow(img_hsv[:, :, 1], cmap='viridis')
    axes[1].set_title('Canal S (Saturación HSV)')
    axes[2].imshow(mag_sobel, cmap='magma')
    axes[2].set_title('Magnitud Gradiente Sobel')
    axes[3].imshow(canny, cmap='gray')
    axes[3].set_title(f'Bordes Canny (Dens: {vector[16]:.3f})')
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# CARGA DEL DATASET
# ==============================================================================
def cargar_dataset(ruta_base):
    imagenes_rgb = []
    X_flatten = []
    X_descriptores = []
    y_labels = []
    nombres_clases = sorted([d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))])
    
    if not nombres_clases:
        raise ValueError(f"No se encontraron subcarpetas de clases en '{ruta_base}'.")

    for idx_clase, clase in enumerate(nombres_clases):
        ruta_clase = os.path.join(ruta_base, clase)
        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_bgr = cv2.imread(os.path.join(ruta_clase, archivo))
                if img_bgr is None:
                    continue
                # Preprocesamiento común: Conversión a RGB y redimensionamiento fijo
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, IMG_SIZE)
                
                vec_desc, _, _, _ = extraer_descriptores(img_resized)
                
                imagenes_rgb.append(img_resized)
                X_flatten.append(img_resized.flatten()) # Estrategia A: 128*128*3 = 49,152 valores
                X_descriptores.append(vec_desc)         # Estrategia C: 17 valores
                y_labels.append(idx_clase)
                
    return np.array(imagenes_rgb), np.array(X_flatten), np.array(X_descriptores), np.array(y_labels), nombres_clases

# ==============================================================================
# EJECUCIÓN PRINCIPAL (ÍTEMS 1.2, 1.3 y 1.4)
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(RUTA_DATASET):
        print(f" Atención: Crea la carpeta '{RUTA_DATASET}' y coloca dentro las subcarpetas de cada clase.")
        exit()

    imgs, X_flat, X_desc, y, clases = cargar_dataset(RUTA_DATASET)
    print(f"Dataset cargado exitosamente: {len(y)} imágenes en {len(clases)} clases: {clases}")

    # 1. Mostrar ejemplo visual y numérico para el Ítem 1.2
    visualizar_ejemplo_descriptores(imgs[0], clases[y[0]])

    # 2. Partición Estratificada y Reproducible del Dataset (70% Train, 30% Test)
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )

    # --- ESTRATEGIA A: IMAGEN ORIGINAL (FLATTEN) ---
    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(X_flat[idx_train])
    X_test_A = scaler_A.transform(X_flat[idx_test])
    
    clf_A = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf_A.fit(X_train_A, y_train)
    y_pred_A = clf_A.predict(X_test_A)

    # --- ESTRATEGIA B: FLATTEN + PCA ---
    # Retenemos el 95% de la varianza explicada para justificar el número de componentes
    pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
    X_train_B = pca.fit_transform(X_train_A)
    X_test_B = pca.transform(X_test_A)
    
    clf_B = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf_B.fit(X_train_B, y_train)
    y_pred_B = clf_B.predict(X_test_B)

    # --- ESTRATEGIA C: DESCRIPTORES DISEÑADOS (17D) ---
    scaler_C = StandardScaler()
    X_train_C = scaler_C.fit_transform(X_desc[idx_train])
    X_test_C = scaler_C.transform(X_desc[idx_test])
    
    clf_C = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf_C.fit(X_train_C, y_train)
    y_pred_C = clf_C.predict(X_test_C)

    # ==============================================================================
    # ÍTEM 1.4: REPORTE DE MÉTRICAS Y REDUCCIÓN DE DIMENSIONALIDAD
    # ==============================================================================
    dim_A = X_train_A.shape[1]
    dim_B = X_train_B.shape[1]
    dim_C = X_train_C.shape[1]

    print("\n" + "="*60)
    print("RESUMEN DE DIMENSIONALIDAD (ÍTEM 1.4)")
    print("="*60)
    print(f"Estrategia A (Flatten)      : {dim_A} características (Referencia)")
    print(f"Estrategia B (PCA 95% var.) : {dim_B} componentes (Reducción del {(1 - dim_B/dim_A)*100:.4f}%)")
    print(f"Estrategia C (Descriptores) : {dim_C} descriptores (Reducción del {(1 - dim_C/dim_A)*100:.4f}%)")

    estrategias = [
        ("Estrategia A: Flatten", y_pred_A),
        (f"Estrategia B: PCA ({dim_B} comp.)", y_pred_B),
        ("Estrategia C: Descriptores (17D)", y_pred_C)
    ]

    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    fig_cm.suptitle('Ítem 1.4: Matrices de Confusión Comparativas', fontsize=16)

    for i, (titulo, y_pred) in enumerate(estrategias):
        print(f"\n--- {titulo} ---")
        print(classification_report(y_test, y_pred, target_names=clases, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
        disp.plot(ax=axes_cm[i], cmap='Blues', colorbar=False)
        axes_cm[i].set_title(titulo)
        axes_cm[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Visualizar ejemplos de aciertos y errores (sobre Estrategia C)
    aciertos = np.where(y_pred_C == y_test)[0]
    errores = np.where(y_pred_C != y_test)[0]

    fig_ej, axes_ej = plt.subplots(1, 2, figsize=(10, 5))
    fig_ej.suptitle('Ejemplos de Clasificación en Prueba (Estrategia C)', fontsize=14)

    if len(aciertos) > 0:
        idx_ok = aciertos[0]
        axes_ej[0].imshow(imgs[idx_test[idx_ok]])
        axes_ej[0].set_title(f"CORRECTO\nReal: {clases[y_test[idx_ok]]}\nPred: {clases[y_pred_C[idx_ok]]}", color='green')
        axes_ej[0].axis('off')

    if len(errores) > 0:
        idx_err = errores[0]
        axes_ej[1].imshow(imgs[idx_test[idx_err]])
        axes_ej[1].set_title(f"INCORRECTO\nReal: {clases[y_test[idx_err]]}\nPred: {clases[y_pred_C[idx_err]]}", color='red')
        axes_ej[1].axis('off')
    else:
        axes_ej[1].text(0.5, 0.5, "¡100% de Aciertos en Test!\nSin errores para mostrar.", ha='center', va='center')
        axes_ej[1].axis('off')

    plt.tight_layout()
    plt.show()
