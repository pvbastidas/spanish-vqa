# Técnicas de optimización de modelos pre-entrenados aplicados a un modelo multimodal en idioma español

Este repositorio contiene el código para explorar distintas técnicas de otimización aplicados a un modelo multimodal imagen-texto en lenguaje español

## Dataset

El dataset corresponde a [DAQUAR](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge) un conjunto de datos para respuestas de preguntas de imágenes del mundo real. Se debera descargar el archivo de texto [Full DAQUAR- all classes](https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.txt) referente a preguntas y respuestas, y tambien el [conjunto de imágenes](http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar) y guardarlos en un repositorio local. 

Debido a que el dataset se encuentra en idioma Ingles es necesario traducirlo al español, para lo cual se usaron dos estrategias; asi para las preguntas se uso el modelo de aprendizaje profundo [nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) y en el caso de las respuestas se uso la libreria de Python [deep-translatio](https://deep-translator.readthedocs.io/en/latest)

Todo el código antes mencionado y un breve resumen estadistico del dataset se lo puede encontrar en [translate_to_spanish_daquar.ipynb](https://github.com/pvbastidas/spanish-vqa/blob/master/dataset_daquar/translate_to_spanish_daquar.ipynb)

## Requerimientos
- `pandas 2.0.2`
- `datasets 2.14.3`
- `scikit-learn 1.2.2`
- `deep-translator 1.11.4`
- `numpy 1.24.3`
- `torch 2.0.1`
- `transformers 4.32.0.dev0`
- `nltk 3.8.1`
- `Pillow 9.5.0`

## Implementación

### ⚡ Pipeline & Scripts
La carpeta `util/` contiene el código base para los experimentos, para experimentar solo cambie la referencia a los modelos de texto e imagen pre-entrenados que desee probar.

Los scripts :

- `util/VisualQuestionAnsweringWithTransformers.ipynb`: Código base Jupyter Notebook 
- `util/modeling_multimodal.py`: Clase del Multimodal Collator
- `dataset_daquar/translate_to_spanish_daquar.ipynb`: Jupyter Notebook para la traducción del dataset DAQUAR al Español

## Modelos

- Algoritmos de Optimización
    - Adam
    - Stochastic Gradient Descent
    - Adafactor
- Transformadores de Texto 
    - BETO: `'bert-base-spanish-wwm-uncased'`
    - BERTIN: `'bertin-roberta-base-spanish'`
    - roBERTA-BASE-BNE:`'roberta-base-bne'
- Transformadores de Imágenes
    - ViT: `'google/vit-base-patch16-224-in21k'`
    - DeiT: `'facebook/deit-base-distilled-patch16-224'`
    - BEiT: `'microsoft/beit-base-patch16-224-pt22k-ft22k'`


## Evaluación

Los mejores resultados del puntaje WUPS se presentan a continuación:

| Transformador Texto | Transformador Imágen | Algoritmo Optimizador | WUPS |
| :---: | :---: | :---: | :---: | 
| ROBERTA-BASE-BNE | BEiT | Adam | _**0.277**_ | 
| BETO | BEiT | Adam | 0.276 | 
| BETO | ViT | Adam | 0.275 | 
| ROBERTA-BASE-BNE | DeiT | Adam | 0.275 |
| BETO | BEiT | Adafactor | 0.275 |





