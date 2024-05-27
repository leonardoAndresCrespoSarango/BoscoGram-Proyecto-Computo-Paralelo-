import os

import cv2
from flask import Flask, request, send_file, render_template
from PIL import Image
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

app = Flask(__name__)

# Cargar la imagen
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_np = np.asarray(image, dtype=np.float32)
    return image_np


# Guardar la imagen
def save_image(image_np, output_path):
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_np)
    image.save(output_path)

# Cargar la imagen usando OpenCV
def load_image_opencv(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error al cargar la imagen")
    return image


# Guardar la imagen usando OpenCV
def save_image_opencv(image_np, output_path):
    cv2.imwrite(output_path, image_np)

# Código CUDA para el efecto vintage
kernel_code = """
__global__ void vintage_filter(float *image, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        float r = image[idx];
        float g = image[idx + 1];
        float b = image[idx + 2];

        // Convertir a escala de grises
        float gray = 0.3f * r + 0.59f * g + 0.11f * b;

        // Aplicar tintes vintage
        image[idx] = gray + 35.0f;       // Red
        image[idx + 1] = gray + 20.0f;   // Green
        image[idx + 2] = gray - 20.0f;   // Blue
    }
}
"""
# Kernel CUDA para efecto de pintura al óleo
oil_paint_kernel = """
__global__ void oil_paint(unsigned char *input, unsigned char *output, int width, int height, int radius, int levels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int histogram[256] = {0};
    int intensity_sum[256][3] = {0};

    for (int ky = -radius; ky <= radius; ky++)
    {
        for (int kx = -radius; kx <= radius; kx++)
        {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            int idx = (py * width + px) * 3;

            int r = input[idx];
            int g = input[idx + 1];
            int b = input[idx + 2];
            int intensity = (r + g + b) / 3;

            histogram[intensity]++;
            intensity_sum[intensity][0] += r;
            intensity_sum[intensity][1] += g;
            intensity_sum[intensity][2] += b;
        }
    }

    int max_intensity = 0;
    for (int i = 1; i < 256; i++)
    {
        if (histogram[i] > histogram[max_intensity])
        {
            max_intensity = i;
        }
    }

    int r = intensity_sum[max_intensity][0] / histogram[max_intensity];
    int g = intensity_sum[max_intensity][1] / histogram[max_intensity];
    int b = intensity_sum[max_intensity][2] / histogram[max_intensity];

    int output_idx = (y * width + x) * 3;
    output[output_idx] = r;
    output[output_idx + 1] = g;
    output[output_idx + 2] = b;
}
"""

# Código CUDA para el círculo con fondo en blanco y negro
circle_kernel_code = """
__global__ void draw_circle_with_background(unsigned char *img, int width, int height, int cx, int cy, int radius) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        int dx = x - cx;
        int dy = y - cy;
        float distance = sqrtf(dx*dx + dy*dy);
        if (distance <= radius) {
            // Mantener el color dentro del círculo
            // No hacer nada, simplemente mantener el color original
        } else {
            // Convertir el área fuera del círculo a blanco y negro
            unsigned char gray_value = (unsigned char)(0.299*img[(y * width + x) * 3] + 0.587*img[(y * width + x) * 3 + 1] + 0.114*img[(y * width + x) * 3 + 2]);
            img[(y * width + x) * 3] = gray_value;
            img[(y * width + x) * 3 + 1] = gray_value;
            img[(y * width + x) * 3 + 2] = gray_value;
        }
    }
}
"""


def apply_oil_painting_effect(image, radius=3, levels=256):
    cuda.init()
    device = cuda.Device(0)  # Asume que estás usando el primer dispositivo CUDA
    context = device.make_context()
    height, width, channels = image.shape

    if channels != 3:
        raise ValueError("La imagen debe tener 3 canales de color (RGB).")

    # Asegurarse de que los datos estén en formato uint8
    image = image.astype(np.uint8)

    # Asignar memoria en la GPU
    input_image_gpu = cuda.mem_alloc(image.nbytes)
    output_image_gpu = cuda.mem_alloc(image.nbytes)

    # Copiar datos a la GPU
    cuda.memcpy_htod(input_image_gpu, image)
    mod = SourceModule(oil_paint_kernel)
    oil_paint = mod.get_function("oil_paint")
    # Ejecutar kernel
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])), 1)

    oil_paint(input_image_gpu, output_image_gpu, np.int32(width), np.int32(height), np.int32(radius), np.int32(levels), block=block_size, grid=grid_size)

    # Crear un array vacío para almacenar la imagen procesada
    output_image = np.empty_like(image)

    # Copiar datos de vuelta a la CPU
    cuda.memcpy_dtoh(output_image, output_image_gpu)
    context.pop()
    return output_image



# Configuración del kernel y ejecución
def apply_vintage_filter(image_np):
    cuda.init()
    device = cuda.Device(0)  # Asume que estás usando el primer dispositivo CUDA
    context = device.make_context()
    height, width, _ = image_np.shape
    image_size = width * height * 3

    # Asignar memoria en la GPU
    image_gpu = cuda.mem_alloc(image_np.nbytes)

    # Copiar datos a la GPU
    cuda.memcpy_htod(image_gpu, image_np)

    # Compilar y ejecutar el kernel
    mod = SourceModule(kernel_code)
    vintage_filter = mod.get_function("vintage_filter")

    # Configuración del bloque e hilo
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(width / 16)), int(np.ceil(height / 16)), 1)

    vintage_filter(image_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    # Copiar el resultado de vuelta a la CPU
    cuda.memcpy_dtoh(image_np, image_gpu)
    # Libera el contexto CUDA
    context.pop()
    return image_np

def apply_circle_with_background_filter(image, radius, cx, cy):
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    height, width, channels = image.shape

    if channels != 3:
        raise ValueError("La imagen debe tener 3 canales de color (RGB).")

    image = image.astype(np.uint8)

    img_gpu = cuda.mem_alloc(image.nbytes)
    cuda.memcpy_htod(img_gpu, image)

    mod = SourceModule(circle_kernel_code)
    draw_circle_with_background_kernel = mod.get_function("draw_circle_with_background")

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

    draw_circle_with_background_kernel(img_gpu, np.int32(width), np.int32(height), np.int32(cx), np.int32(cy), np.int32(radius), block=block, grid=grid)

    result = np.empty_like(image)
    cuda.memcpy_dtoh(result, img_gpu)
    context.pop()
    return result

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        # Cargar y procesar la imagen
        image_np = load_image(file_path)
        image_np_vintage = apply_vintage_filter(image_np)

        # Guardar la imagen procesada
        output_path = os.path.join('static/outputs', 'vintage_' + filename)
        save_image(image_np_vintage, output_path)

        return send_file(output_path, mimetype='image/jpeg')

@app.route('/upload/oil', methods=['POST'])
def upload_image_oil():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        # Cargar y procesar la imagen
        image_np = load_image(file_path)
        output_image = apply_oil_painting_effect(image_np)


        # Guardar la imagen procesada
        output_path = os.path.join('static/outputs', 'oil_' + filename)
        save_image(output_image, output_path)

        return send_file(output_path, mimetype='image/jpeg')

@app.route('/upload/circle', methods=['POST'])
def upload_image_circle():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        image_np = load_image_opencv(file_path)

        height, width, _ = image_np.shape
        radius = 100
        cx, cy = width // 2, height // 2

        output_image = apply_circle_with_background_filter(image_np, radius, cx, cy)

        output_path = os.path.join('static/outputs', 'circleFilter_' + filename)
        save_image_opencv(output_image, output_path)

        return send_file(output_path, mimetype='image/jpeg')


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=int("5000"), debug=True,ssl_context='adhoc')

#ngrok http --domain=penguin-healthy-iguana.ngrok-free.app 5000