
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

![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/1.png)

Estas funciones permiten cargar y guardar imágenes tanto con Pillow como con OpenCV.

#### Filtro Vintage
Como primer filtro hemos optado por un efecto vintage, para lo cual se tiene que definir el kernel pasando 3 parámetros, el ancho y alto de la imagen, además de un puntero image que contendrá la dirección de memoria de otra variable. 

Posteriormente debemos crear las variables x e y, en donde se calcularán las coordenadas globales de x e y del hilo actual dentro la imagen que se está procesando. Luego tenemos una condicional en donde se asegura que solo los hilos que corresponden a pixeles dentro de los límites de la imagen realizarán su respectivo procesamiento. De igual manera tenemos un índice idx correspondiente a la posición en el arreglo image donde se almacenarán los valores de color RBG del píxel en las coordenadas(x,y), y debemos multiplicar por 3 ya que se tiene 3 canales. Luego definimos 3 variables flotantes r,g,b las cuales obtendrán el valor correspondiente al respectivo índice en los 3 canales, posteriormente convertiremos cada uno de los canales a escala de grises, en donde multiplicamos el valor del índice por una ponderación adecuada, en donde se debe dar mayor ponderación al verde ya que este canal tiene mayor influencia en la percepción del brillo. Finalmente, para aplicar el filtro debemos pasar nuestra variable gray para cada canal, a la cual se le sumará diferentes valores que harán que nuestra imagen tenga un color amarillento. 


![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/2.png)
1. Identificación de la Posición del Pixel:

- int x = blockIdx.x * blockDim.x + threadIdx.x;
- int y = blockIdx.y * blockDim.y + threadIdx.y;
- blockIdx, blockDim y threadIdx
- son propiedades CUDA que indican la posición del bloque y el hilo dentro de ese bloque. Estas líneas calculan las coordenadas (x, y) del pixel que se está procesando.

2. Verificación de Límites:

- if (x < width && y < height) {
- Se asegura de que las coordenadas (x, y) estén dentro de los límites de la imagen.
  
3. Cálculo del Índice del Pixel:

- int idx = (y * width + x) * 3;
- Calcula el índice del pixel en la matriz unidimensional. Cada pixel tiene 3 valores (r, g, b), por eso se multiplica por 3.
  
4. Extracción de los Valores RGB:

- float r = image[idx];
- float g = image[idx + 1];
- float b = image[idx + 2];
- Se extraen los valores de los componentes rojo, verde y azul del pixel.
  
5. Conversión a Escala de Grises:

- float gray = 0.3f * r + 0.59f * g + 0.11f * b;
- Convierte el pixel a un valor en escala de grises usando una fórmula ponderada comúnmente usada en el procesamiento de imágenes.
  
6. Aplicación de Tinte Vintage:

- image[idx] = gray + 35.0f;
- image[idx + 1] = gray + 20.0f;
- image[idx + 2] = gray - 20.0f;
- Ajusta los valores de gris para cada canal de color para darle un efecto vintage. El canal rojo se incrementa más, seguido por el verde y finalmente el azul.




