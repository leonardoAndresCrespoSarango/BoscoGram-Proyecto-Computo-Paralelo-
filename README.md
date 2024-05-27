
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

![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/3.png)

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
    - Se asegura que las coordenadas del vecindario estén dentro de los límites de la imagen.
    - Extrae los valores RGB y calcula la intensidad media.
    - histogram[intensity]++;
    - intensity_sum[intensity][0] += r;
    - intensity_sum[intensity][1] += g;
    - intensity_sum[intensity][2] += b;
    - Actualiza el histograma y las sumas de intensidad.
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
#### Filtro Circulo ocn Fondo blanco y negro
Finalmente tenemos el tercer filtro, en donde se dibuja un círculo en el centro y todos los píxeles dentro del círculo mantendrán su color orginal y los píxeles de fuera estarán en blanco y negro. Para este filtro pasamos un puntero img, el ancho y alto de la imagen, así también se debe pasar las coordenadas del centro del círculo y la varibale raidus, es decir el radio del círculo. 

Una vez mas tenemos las variables x e y, en donde se calculan las coordenadas globales del píxel actual, una vez más tenemos una condicional que verificará que el píxel actual se encuentre dentro de los límites de la imagen, dentro de esta tenemos las variables dx y dy que se encargan de calcular las diferencias entre las coordenadas x y y del píxel actual y las coordenadas cx y cy del centro del círculo, y para calculor la distancia deberemos usar el teorema de Pitágoras en donde aplicaremos la raíz cuadrada de dx y dy. Posteriormente tenemos la condicional que verifica si la distancia es menor o igual al radio, significa que el píxel está dentro del círculo por lo que no se realizará nada, en cambio si los pixeles están fuera del círculo serán convertidos a blanco y negro para ello tenemos la variable gray_value que almacenará el valor de gris del píxel fuera del círculo, finalmente las últimas 3 líneas de código calculan el valor de gris del píxel utilizando una combinación ponderada de sus componentes de color y luego asignan este valor de gris a todos los componentes de color del píxel, convirtiendo así el color del píxel a blanco y negro. 

![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/4.png)
1. Identificación de la Posición del Pixel:

- int x = threadIdx.x + blockIdx.x * blockDim.x;
- int y = threadIdx.y + blockIdx.y * blockDim.y;
2. Verificación de Límites:

- if (x < width && y < height) {
3. Cálculo de la Distancia al Centro del Círculo:

- int dx = x - cx;
- int dy = y - cy;
- float distance = sqrtf(dx*dx + dy*dy);
  Calcula la distancia entre el pixel actual y el centro del círculo usando el teorema de Pitágoras.
4. Condición Dentro/Fuera del Círculo:

- if (distance <= radius) {
  Si el pixel está dentro del radio del círculo, mantiene el color original.
5. Conversión a Blanco y Negro:

- unsigned char gray_value = (unsigned char)(0.299*img[(y * width + x) * 3] + 0.587*img[(y * width + x) * 3 + 1] + 0.114*img[(y * width + x) * 3 + 2]);
- img[(y * width + x) * 3] = gray_value;
- img[(y * width + x) * 3 + 1] = gray_value;
- img[(y * width + x) * 3 + 2] = gray_value;
  Si el pixel está fuera del círculo, convierte el color a escala de grises usando la fórmula ponderada común y asigna el valor gris a todos los canales RGB.

## Definición de Funciones
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/5.png)
Una vez creados los 3 filtros, debemos definir las funciones que aplicarán cada uno de estos filtros. 

En donde primero definimos apply_vintage_filter,y pasamos la imagen, de igual manera deberemos iniciar cuda, luego de ello, calculamos el  ancho y alto con el .shape. Posteriormente enimage_gpu debemos asignar la memmoria en la GPU, y luego copiar los datos a la gpu. Luego se debe compilar y ejecutar el kernel además que  se debe realiar la configuracion del bloque e hilo: block_size = (16, 16, 1) 
grid_size = (int(np.ceil(width / 16)), int(np.ceil(height / 16)), 1) en donde pasamos 16 hilos para x e y, y para x 1 hilo. Luego el tamaño del bloque se calculará automaciamente. Finalmente copiamos el resultado a la cpu. 

De igual manera definimos  apply_circle apply_oil_painting_effect  

_with_background_filter, pero en este caso pasamos un radio de 3 y levels de 256. Y realizamos los mismos pasos que el anterior método, a diferencia que al compilar el mpetodo se debe pasar los valores de radius y levels.  

 

De la misma manera se define el metodo apply_circle_with_background_filter en donde pasamos la imagen, el radius y las cooredenadas del centro en x e y,  y se realizamos los mismos pasos especificados en los métodos anteriores. 


### Funciones de Filtro

#### `apply_vintage_filter(image_np)`

1. Inicializa CUDA.
2. Calcula el ancho y alto de la imagen.
3. Asigna memoria en la GPU y copia los datos.
4. Compila y ejecuta el kernel.
5. Configura el tamaño del bloque y la cuadrícula.
6. Copia el resultado a la CPU.

#### `apply_oil_painting_effect(image, radius, levels)`

1. Inicializa CUDA.
2. Calcula el ancho y alto de la imagen.
3. Asigna memoria en la GPU y copia los datos.
4. Compila y ejecuta el kernel.
5. Configura el tamaño del bloque y la cuadrícula.
6. Copia el resultado a la CPU.

#### `apply_circle_with_background_filter(image, radius, cx, cy)`

1. Inicializa CUDA.
2. Calcula el ancho y alto de la imagen.
3. Asigna memoria en la GPU y copia los datos.
4. Compila y ejecuta el kernel.
5. Configura el tamaño del bloque y la cuadrícula.
6. Copia el resultado a la CPU.

## Rutas de la Aplicación Flask

- **`/upload/image`**: Carga una imagen y aplica el filtro vintage.
- **`/upload/oil`**: Carga una imagen y aplica el efecto de pintura al óleo.
- **`/upload/circle`**: Carga una imagen y aplica el filtro de círculo con fondo en blanco y negro.
- ![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/6.png)
## Ejecución del Servidor
Una vez definidos los 3 métodos más que cargarán, procesarán las imágenes y guardarán el resultado final. Finalmente iniciamos un servidor Flask que será expuesto en el puerto 5000 y tendrá soporte SSL.
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/7.png)
Nota: Hay que tener en cuenta que en nuestro caso optamos por utilizar NGROK que es un software locar para crear tuneles SSL (en pocas palabras  crear una ruta https segura para la comunicación entre ambas partes).
Obtamos por utilizar un dominio statico, mas que nada para  obviar la parte de colocar un cuadro de texto en donde el usuario tenga que colocar la ip correspondiente al servidor: 
`ngrok http --domain=penguin-healthy-iguana.ngrok-free.app 5000` .
Como podemos observar basta con colcoar el puerto del servidor ya nuestro dominio sera estatico y asi facilitar el uso por parte del usuario.
### Dockerfile
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/dockerfile.png)
Este archivo define la configuración del entorno Docker necesario para ejecutar la aplicación.

#### Pasos Principales

1. Utiliza la imagen base `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
2. Configura el directorio de trabajo y actualiza los paquetes.
3. Instala las dependencias necesarias, incluyendo `pycuda` y otras bibliotecas de Python.
4. Copia los archivos del proyecto al contenedor.
5. Expone el puerto 5000 y define el comando para ejecutar la aplicación Flask.

## Ejecución de la Aplicación

### Requisitos Previos

- Docker instalado en el sistema.
- GPU compatible con CUDA.

### Construcción y Ejecución con Docker

1. Construir la imagen Docker:

    ```sh
    docker build -t flask-cuda-app .
    ```

2. Ejecutar el contenedor Docker:

    ```sh
    docker run --gpus all -p 5000:5000 flask-cuda-app
    ```

### Uso de la Aplicación

1. Acceder a la aplicación web en `https://localhost:5000` (en caso de optar por utilizar la URL que ofrece Docker requiere modificacio URL en FrontEnd).
1.1. Acceder directamente desde `https://penguin-healthy-iguana.ngrok-free.app` que es nuestro tunel SSL (Dominio estatico).
2. Crear usuario o Iniciar sesion con las credenciales ya creadas.
3. Cargar una imagen desde la interfaz.
4. Seleccionar el filtro deseado (vintage, pintura al óleo o círculo con fondo en blanco y negro).
5. Descargar la imagen procesada.
6. Publicar la iamgen descargada
## Resultados
### Filtro Efecto Vintage
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/filtro vintage original.webp)
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/filtro vintage.webp)
### Filtro Efecto Pintura de oleo
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/filtro oleo original.jpg)
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/filtro oleo.jpg)
### Filtro circulo con fondo blanco y negro
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/filtro circulo original.jpg)
![Logo](https://github.com/leonardoAndresCrespoSarango/BoscoGram-Servidor/blob/1ed283c00b7a894b34eeb11f82bd3c4b0f83a5b9/imagenes/filtro circulo.jpg)

## Conclusiones

Este proyecto demuestra cómo combinar Flask y CUDA para crear una aplicación web eficiente que aplica filtros de imágenes utilizando el procesamiento paralelo de la GPU. Docker facilita la configuración del entorno, asegurando que las dependencias y configuraciones necesarias estén disponibles de manera consistente.

### Posibles Mejoras

- Implementar una interfaz de usuario más amigable y atractiva.
- Agregar más filtros y opciones de procesamiento de imágenes.
- Optimizar aún más los kernels de CUDA para mejorar el rendimiento.
- Incluir pruebas automatizadas para asegurar la robustez y calidad del código.
