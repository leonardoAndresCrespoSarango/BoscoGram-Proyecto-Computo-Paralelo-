
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
  son propiedades CUDA que indican la posición del bloque y el hilo dentro de ese bloque. Estas líneas calculan las coordenadas (x, y) del pixel que se está procesando.

2. Verificación de Límites:

- if (x < width && y < height) {
  Se asegura de que las coordenadas (x, y) estén dentro de los límites de la imagen.
  
3. Cálculo del Índice del Pixel:

- int idx = (y * width + x) * 3;
  Calcula el índice del pixel en la matriz unidimensional. Cada pixel tiene 3 valores (r, g, b), por eso se multiplica por 3.
  
4. Extracción de los Valores RGB:

- float r = image[idx];
- float g = image[idx + 1];
- float b = image[idx + 2];
  Se extraen los valores de los componentes rojo, verde y azul del pixel.
  
5. Conversión a Escala de Grises:

- float gray = 0.3f * r + 0.59f * g + 0.11f * b;
  Convierte el pixel a un valor en escala de grises usando una fórmula ponderada comúnmente usada en el procesamiento de imágenes.
  
6. Aplicación de Tinte Vintage:

- image[idx] = gray + 35.0f;
- image[idx + 1] = gray + 20.0f;
- image[idx + 2] = gray - 20.0f;
  Ajusta los valores de gris para cada canal de color para darle un efecto vintage. El canal rojo se incrementa más, seguido por el verde y finalmente el azul.

  #### Filtro Efecto pintura de oleo
Como segundo filtro tenemos el efecto de pintura al óleo, en donde pasareos 2 puntos input y output que representarán la imagen de entrada y salida, pasamos el ancho y alto de la imagen, el radio que indica la extensión del área alrededor de cada pixél que será considerado para aplicar el efecto, además incluimos la variable levels que indicará la intensidad del efecto. 

Luego una vez más tenemos las variables respectivas para el cálculo de las coordenadas globales del píxel que el hilo irá a procesar. Posterior a ello tenemos una condicional que verifica que los pixeles se encuentren dentro de los límites de la imagen. Dentro de la condicional debemos iniciar la variable histogram que contará la cantidad de pixeles con cada intensidad de color, y la variable intensity_sum almacenará la suma de los valores de cada componente de color para cada intensidad de color en la imagen.  

Luego tendremos un bucle for que recorrerá todos los píxeles dentro de una vecindad centrada en el píxel actual. Para ello pasamos nuestra variable radius que determina cuantos píxeles hacia arriba, abajo a la izquierda y derecha se considerarán. Dentro de este bucle tenemos las variables px y py que calcularán las coordenadas del píxel vecino dentro del radio indicado, y una vez al índice idx se lo multiplicará por 3 debido a los 3 canales correspondientes. De igual manera tenemos las 3 variables r,g,b que obtendrán el valor del píxel en cada canal, además tenemos la variable intensity que es calculada por el promedio de las tres variables r,g,b, la cual proporciona una estimación de la cantidad total de "color" en el píxel, que se usa para actualizar el histograma y las sumas de intensidades.  luego deberemos actualizar las sumas de intensidades sumando los valores de cada componente de color a las respectivas sumas de intensidades para la intensidad calculada. 

Seguidamente, tenemos un bucle que busca el valor de intensidad más frecuente en el histograma asignándola en la variable max_intensity. Con ello Se calcula el color promedio para la intensidad más frecuente utilizando las sumas de intensidades y el histograma, finalmente se realiza la asignación del color promedio al píxel de salida. Esto crea un efecto similar al de una pintura al óleo, donde los detalles finos se simplifican en áreas de color más uniforme. 

![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/2.png)

1. Identificación de la Posición del Pixel:

- int x = blockIdx.x * blockDim.x + threadIdx.x;
- int y = blockIdx.y * blockDim.y + threadIdx.y;
2. Verificación de Límites:

- if (x >= width || y >= height) return;
  Si las coordenadas están fuera de los límites, el hilo termina.
3. Inicialización de Histogramas e Intensity Sums:

- int histogram[256] = {0};
- int intensity_sum[256][3] = {0};
  Arrays para almacenar la frecuencia de las intensidades y la suma de colores para cada intensidad.
4. Cálculo del Histograma Local:

- Itera sobre un vecindario definido por el radio (radius) alrededor del pixel actual.
- for (int ky = -radius; ky <= radius; ky++)
- for (int kx = -radius; kx <= radius; kx++)
    - Para cada pixel en el vecindario:
    - int px = min(max(x + kx, 0), width - 1);
    - int py = min(max(y + ky, 0), height - 1);
      Se asegura que las coordenadas del vecindario estén dentro de los límites de la imagen.
          - Extrae los valores RGB y calcula la intensidad media.
          - histogram[intensity]++;
          - intensity_sum[intensity][0] += r;
          - intensity_sum[intensity][1] += g;
          - intensity_sum[intensity][2] += b;
            Actualiza el histograma y las sumas de intensidad.
5. Determinación de la Intensidad Más Frecuente:

- int max_intensity = 0;
- for (int i = 1; i < 256; i++)
  Encuentra la intensidad con la mayor frecuencia en el histograma.
6. Calculo del Color Promedio:

- int r = intensity_sum[max_intensity][0] / histogram[max_intensity];
- int g = intensity_sum[max_intensity][1] / histogram[max_intensity];
- int b = intensity_sum[max_intensity][2] / histogram[max_intensity];
  Calcula el color promedio de la intensidad más frecuente.
7. Asignación del Nuevo Color:

- int output_idx = (y * width + x) * 3;
- output[output_idx] = r;
- output[output_idx + 1] = g;
- output[output_idx + 2] = b;
  Asigna los valores de color calculados al pixel de salida.


