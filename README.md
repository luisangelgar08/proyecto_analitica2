# Proyecto Analítica 2  
## Router + Mixture of Experts para clasificación médica heterogénea

**Autor:** Luis Angel Garcia  
**Repositorio:** `https://github.com/luisangelgar08/proyecto_analitica2`

---

## 1. Descripción general

Este proyecto implementa una arquitectura **Mixture of Experts (MoE)** para clasificación médica heterogénea.  
El sistema recibe una **imagen 2D** o un **volumen 3D**, aplica un **preprocesado adaptativo**, extrae una representación común con un **Vision Transformer compartido y congelado**, y un **router visual** decide qué experto especializado debe activarse.

La solución integra cinco expertos médicos:

- **NIH Chest X-ray** — radiografías torácicas multilabel
- **ISIC** — dermatoscopía multiclase
- **Osteoarthritis** — clasificación de severidad KL en radiografías de rodilla
- **LUNA16** — clasificación binaria en CT pulmonar 3D
- **Pancreatic Cancer / PANORAMA** — clasificación binaria en CT abdominal 3D

Además del sistema MoE, el proyecto incluye:

- ablation study del router
- auxiliary loss de balance de carga
- detección OOD por entropía del router
- dashboard de inferencia
- evaluación end-to-end del sistema

---

## 2. Objetivo

Construir un sistema MoE capaz de:

1. recibir datos médicos heterogéneos 2D y 3D,
2. enrutar automáticamente cada muestra al experto correcto,
3. mantener balance de carga entre expertos,
4. detectar entradas fuera de distribución,
5. operar end-to-end dentro de restricciones reales de hardware.

---

## 3. Arquitectura general

### Flujo del sistema

**Entrada heterogénea**  
→ **Preprocesado adaptativo 2D/3D**  
→ **Backbone ViT compartido y congelado**  
→ **Embedding común (192D)**  
→ **Router final ViT + Linear**  
→ **Experto especializado activado**  
→ **Predicción clínica final**

### Componentes principales

- **Preprocesado adaptativo**
  - detección automática 2D/3D
  - resize, normalización y tensorización
- **Backbone del router**
  - Vision Transformer compartido
  - embeddings congelados de 192 dimensiones
- **Router final**
  - capa Linear entrenada sobre embeddings ViT
  - auxiliary loss con `alpha = 0.02`
  - embeddings estandarizados con scaler guardado
- **Expertos**
  - modelos especializados por dominio médico
- **Dashboard**
  - inferencia
  - heatmap aproximado del router
  - tabla del ablation
  - balance de carga
  - detección OOD

---

## 4. Expertos del sistema

| Experto | Tipo | Modelo base | Estado final |
|---|---|---:|---|
| NIH Chest X-ray | 2D multilabel | DenseNet121 | Provisional |
| ISIC | 2D multiclase | EfficientNet-B3 | Usable provisional |
| Osteoarthritis | 2D multiclase | ResNet34 | Más sólido |
| LUNA16 | 3D binario | R3D-18 | Aceptable provisional |
| Pancreatic Cancer | 3D binario | R3D-18 | Aceptable provisional |

---

## 5. Ablation study del router

El router se comparó sobre los **mismos embeddings ViT congelados**, usando cuatro mecanismos:

- **Linear (ViT + capa linear)**
- **k-NN (FAISS)**
- **Gaussian Naive Bayes**
- **GMM**

### Resultados del ablation

| Router | Val Acc. | Test Acc. | Latencia (ms/sample) | Lectura |
|---|---:|---:|---:|---|
| Linear | 0.9976 | 0.9976 | 0.0008 | Ganador |
| k-NN | 0.9927 | 0.9855 | 0.0235 | Muy competitivo |
| Gaussian Naive Bayes | 0.9146 | 0.9229 | 0.0078 | Baseline rápido |
| GMM | 0.7512 | 0.7663 | 0.0060 | No alcanzó el objetivo |

El router **Linear** fue seleccionado como versión final por su mejor precisión, estabilidad e integración con el MoE.

---

## 6. Métricas sistémicas finales

### Router final

- **Routing Accuracy (test):** `0.9976`
- **Best epoch:** `48`
- **Auxiliary loss alpha:** `0.02`
- **Load balance ratio:** `1.0244`

### Balance de carga final

- `f_i NIH = 0.2000`
- `f_i ISIC = 0.2000`
- `f_i Osteo = 0.2000`
- `f_i LUNA16 = 0.1976`
- `f_i Pancreas = 0.2024`

### Sistema MoE completo

- **Routing accuracy overall:** `0.9976`
- **Inferencias exitosas:** `415 / 415`
- **Errores técnicos:** `0`

### OOD detection

- **OOD AUROC:** `0.9762`
- **Umbral de entropía p95(ID):** `0.4031`
- **Entropía media ID:** `0.0876`
- **Entropía media OOD:** `0.7817`
- **ID alert rate:** `0.0500`
- **OOD alert rate:** `0.8700`

### Rendimiento computacional

- **Router only - Pico VRAM reservada:** `0.451 GB`
- **MoE full - Pico VRAM reservada:** `0.584 GB`
- **Latencia media router only:** `2109.40 ms`
- **Latencia media MoE full:** `2788.26 ms`

---

## 7. Estructura real del repositorio

```text
proyecto_analitica2/
│
├── data/
│   └── manifests/
│       ├── manifest_isic_master.csv
│       ├── manifest_luna16.csv
│       ├── manifest_nih_final.csv
│       ├── manifest_nih_final_with_paths.csv
│       ├── manifest_nih_master.csv
│       ├── manifest_pancreas.csv
│       ├── osteo_master_manifest.csv
│       ├── osteoarthritis_master_manifest.csv
│       ├── patient_split_pancreas.csv
│       ├── split_final_luna16.csv
│       ├── split_final_pancreas.csv
│       └── subset_summary_luna16.csv
│
├── notebooks/
│   ├── debug/
│   ├── isic/
│   ├── luna16/
│   ├── moe/
│   ├── nih/
│   ├── osteo/
│   ├── pancreas/
│   └── router/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 8. Organización del proyecto

- `data/manifests/`: manifiestos consolidados, splits finales y CSV auxiliares de los datasets.
- `notebooks/nih/`: notebooks del experto NIH.
- `notebooks/isic/`: notebooks del experto ISIC.
- `notebooks/osteo/`: notebooks del experto Osteoarthritis.
- `notebooks/luna16/`: notebooks del experto LUNA16.
- `notebooks/pancreas/`: notebooks del experto Pancreatic Cancer / PANORAMA.
- `notebooks/router/`: extracción de embeddings, ablation study y entrenamiento del router.
- `notebooks/moe/`: integración, evaluación y análisis del sistema MoE completo.
- `notebooks/debug/`: pruebas auxiliares y depuración.

---

## 9. Requisitos del entorno

### Hardware usado

- Clúster universitario
- `2 × NVIDIA TITAN Xp (12 GB VRAM c/u)`
- `48 CPUs lógicos`
- `125 GiB RAM`

### Entorno

- Python 3
- `.venv`
- instalación con `pip`
- ejecución por SSH / VS Code Remote

---

## 10. Instalación

Clonar el repositorio:

```bash
git clone https://github.com/luisangelgar08/proyecto_analitica2.git
cd proyecto_analitica2
```

Crear entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 11. Flujo de trabajo recomendado

### 1. Explorar y validar manifests
Revisar los CSV dentro de `data/manifests/` para verificar rutas, splits y etiquetas.

### 2. Ejecutar notebooks por experto
Correr los notebooks de cada carpeta de `notebooks/` según el dominio:

- `notebooks/nih/`
- `notebooks/isic/`
- `notebooks/osteo/`
- `notebooks/luna16/`
- `notebooks/pancreas/`

### 3. Ejecutar notebooks del router
Usar `notebooks/router/` para:
- construir dataset balanceado del router,
- extraer embeddings ViT,
- ejecutar el ablation study,
- entrenar el router final con auxiliary loss.

### 4. Ejecutar notebooks del MoE
Usar `notebooks/moe/` para:
- integrar router + expertos,
- evaluar el sistema completo,
- revisar métricas finales y OOD.

---

## 12. Reproducibilidad

Para mantener reproducibilidad del proyecto:

- fijar seeds cuando aplique,
- conservar por separado:
  - checkpoints finales de expertos,
  - scaler del router,
  - checkpoint del router final,
  - métricas por split,
- no sobrescribir checkpoints buenos durante pruebas posteriores,
- mantener sincronía entre scaler de entrenamiento e inferencia en el router.

---

## 13. Decisiones metodológicas clave

- No se usaron **metadatos externos** de modalidad o dataset como entrada.
- El router se entrenó sobre un **dataset balanceado por experto**.
- El ablation se hizo sobre **embeddings congelados comunes**.
- El principal problema del routing fue la frontera **Pancreas vs LUNA16**.
- La corrección final dependió de:
  - **auxiliary loss**
  - uso consistente del **scaler**
  - checkpoint correcto del experto Pancreas
- El dashboard forma parte del sistema final.

---

## 14. Estado final

El sistema quedó funcional **end-to-end**:

- router final entrenado y evaluado,
- MoE integrado,
- balance de carga correcto,
- detección OOD funcional,
- inferencia exitosa sobre el conjunto balanceado de test.

---

## 15. Referencias principales

```text
[1] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers,
“ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on
Weakly-Supervised Classification and Localization of Common Thorax Diseases,”
in Proc. CVPR, 2017.

[2] ISIC Archive, “ISIC 2019 Challenge Dataset,” 2019.

[3] N. C. Codella et al., “Skin Lesion Analysis Toward Melanoma Detection 2018:
A Challenge Hosted by the International Skin Imaging Collaboration (ISIC),”
arXiv:1902.03368, 2019.

[4] S. G. Armato III et al., “The Lung Image Database Consortium (LIDC) and
Image Database Resource Initiative (IDRI): A Completed Reference Database of
Lung Nodules on CT Scans,” Medical Physics, vol. 38, no. 2, pp. 915–931, 2011.

[5] LUNA16 Grand Challenge, “Data – LUNA16,” 2016.

[6] P. Chen, “Knee Osteoarthritis Severity Grading Dataset,” Mendeley Data,
v1, 2018, doi: 10.17632/56rmx5bjcr.1.

[7] J. H. Kellgren and J. S. Lawrence, “Radiological Assessment of Osteo-Arthrosis,”
Annals of the Rheumatic Diseases, vol. 16, no. 4, pp. 494–502, 1957.

[8] PANORAMA, “Public Training and Development Dataset,” Zenodo, 2024.

[9] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton,
“Adaptive Mixtures of Local Experts,” Neural Computation, vol. 3, no. 1,
pp. 79–87, 1991.

[10] A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for
Image Recognition at Scale,” ICLR, 2021.

[11] W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to
Trillion Parameter Models with Simple and Efficient Sparsity,”
Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.
```

---

## 16. Autor

**Luis Angel Garcia**  
Proyecto Analítica 2 — Router + Mixture of Experts para clasificación médica heterogénea
