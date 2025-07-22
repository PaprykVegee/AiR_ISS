import cv2
import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt

# Kod OpenCL dla erozji
kernel_code = """
const int kernel_size = 3;
const int halfSize = kernel_size / 2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void erode_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float threshold = 100.0;
    bool wynik_erozji = 1;

    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            int2 sample_coord = coord + (int2)(i, j);

            if (sample_coord.x >= 0 && sample_coord.x < get_image_width(inputImage) &&
                sample_coord.y >= 0 && sample_coord.y < get_image_height(inputImage)) {
                
                uint4 pixel = read_imageui(inputImage, imageSampler, sample_coord);
                float grayValue = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
                bool binaryPixel = step(threshold, grayValue);
                wynik_erozji &= binaryPixel;
            }
        }
    }
    
    uint4 result_pixel = (uint4)(255 * wynik_erozji, 255 * wynik_erozji, 255 * wynik_erozji, 255);
    write_imageui(outputImage, coord, result_pixel);
}
"""

# Wczytaj obraz
input_image = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)

# Zmienna do przechowywania czasów dla różnych iteracji
opencv_times = []
opencl_times = []

# OpenCL setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Załaduj kod OpenCL
program = cl.Program(context, kernel_code).build()

# Wczytaj obraz i przygotuj dane wejściowe
input_image = np.array(input_image, dtype=np.uint8)

# Tworzenie buforów OpenCL
input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_image)
output_image = np.zeros_like(input_image)
output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output_image.nbytes)

# Testowanie różnych liczby iteracji
for iterations in range(1, 101):
    # OpenCV
    start_time_opencv = time.time()
    for _ in range(iterations):
        eroded_image_opencv = cv2.erode(input_image, kernel, iterations=1)
    opencv_time = time.time() - start_time_opencv
    opencv_times.append(opencv_time)

    # OpenCL
    start_time_opencl = time.time()
    for _ in range(iterations):
        program.erode_filter(queue, input_image.shape, None, input_buffer, output_buffer)
    cl.enqueue_copy(queue, output_image, output_buffer).wait()
    opencl_time = time.time() - start_time_opencl
    opencl_times.append(opencl_time)

# Tworzenie wykresu
plt.plot(range(1, 101), opencv_times, label='OpenCV')
plt.plot(range(1, 101), opencl_times, label='OpenCL')

# Konfigurowanie wykresu
plt.xlabel('Liczba iteracji')
plt.ylabel('Czas (sekundy)')
plt.title('Porównanie czasów erozji: OpenCV vs OpenCL')
plt.legend()
plt.grid(True)

# Wyświetlanie wykresu
plt.show()
