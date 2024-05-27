
![Logo](https://upload.wikimedia.org/wikipedia/commons/b/b0/Logo_Universidad_Polit%C3%A9cnica_Salesiana_del_Ecuador.png)


# Informe de Práctica: Desarrollo de un Servidor Flask con Procesamiento de Imágenes usando CUDA



## Authors

- [Leonardo Crespo](https://github.com/leonardoAndresCrespoSarango)
- [Carlos Saico]()

## Descripción General

Este proyecto consiste en el desarrollo de una aplicación móvil como FrontEnd Android Studio, utilizando Flask como servidor y aplicando Firebase como base de datos para la gestión de usuarios, la cual permite aplicar varios filtros a imágenes cargadas por el usuario. Los filtros implementados incluyen un efecto vintage, un efecto de pintura al óleo y un filtro de círculo con fondo en blanco y negro. El procesamiento de imágenes se realiza utilizando CUDA para aprovechar el paralelismo de la GPU y acelerar las operaciones.

## Estructura del Proyecto

### Archivos Principales (Servidor)

1. **app.py**: Contiene el código principal de la aplicación Flask y las funciones de procesamiento de imágenes.
2. **Dockerfile**: Define el entorno de Docker necesario para ejecutar la aplicación, incluyendo la instalación de CUDA y las dependencias de Python.
3. **requirements.txt**: Lista las dependencias de Python necesarias para el proyecto.

### app.py

Este archivo configura la aplicación Flask y define varias rutas para manejar la carga y el procesamiento de imágenes.

#### Funciones Principales

- **load_image(image_path)**: Carga una imagen desde la ruta especificada y la convierte a un arreglo NumPy.
- **save_image(image_np, output_path)**: Guarda un arreglo NumPy como una imagen en la ruta especificada.
- **load_image_opencv(image_path)**: Carga una imagen usando OpenCV.
- **save_image_opencv(image_np, output_path)**: Guarda una imagen usando OpenCV.
- **apply_vintage_filter(image_np)**: Aplica un filtro vintage a la imagen usando CUDA.
- **apply_oil_painting_effect(image, radius, levels)**: Aplica un efecto de pintura al óleo a la imagen usando CUDA.
- **apply_circle_with_background_filter(image, radius, cx, cy)**: Aplica un filtro de círculo con fondo en blanco y negro usando CUDA.

## Descripción de los Filtros

### Importación de Módulos

El archivo comienza con la importación de varios módulos necesarios para la funcionalidad de la aplicación, como os para las operaciones del sistema de archivos, cv2 para la manipulación de imágenes con OpenCV, Flask para la creación de la aplicación web, PIL para la manipulación de imágenes con Pillow, numpy para el manejo de matrices numéricas, y pycuda para la programación en GPU usando CUDA.

#### Definición del Kernel

El kernel de CUDA para el filtro vintage toma tres parámetros:
- **width**: Ancho de la imagen.
- **height**: Alto de la imagen.
- **image**: Puntero que contiene la dirección de memoria de la imagen.

#### Funciones de Utilidad para Cargar y Guardar Imágenes





