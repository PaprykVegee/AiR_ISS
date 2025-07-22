#define KERNEL_WIDTH 5           // Szerokość filtra
#define KERNEL_HEIGHT 5          // Wysokość filtra
#define KERNEL_STD 1.0f          // Odchylenie standardowe (std)

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    // Pozycja piksela w obrazie
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    // Współczynniki filtra Gaussa
    float gaussian_kernel[KERNEL_HEIGHT][KERNEL_WIDTH];
    float gaussian_sum = 0.0f;

    // Obliczanie filtra Gaussa
    int kernel_half_x = KERNEL_WIDTH / 2;
    int kernel_half_y = KERNEL_HEIGHT / 2;
    
    for (int y = -kernel_half_y; y <= kernel_half_y; y++) {
        for (int x = -kernel_half_x; x <= kernel_half_x; x++) {
            gaussian_kernel[y + kernel_half_y][x + kernel_half_x] = 
                exp(-((x * x + y * y) / (2.0f * KERNEL_STD * KERNEL_STD))) / (2.0f * M_PI * KERNEL_STD * KERNEL_STD);
            gaussian_sum += gaussian_kernel[y + kernel_half_y][x + kernel_half_x];
        }
    }

    // Normalizacja filtra Gaussa
    for (int y = 0; y < KERNEL_HEIGHT; y++) {
        for (int x = 0; x < KERNEL_WIDTH; x++) {
            gaussian_kernel[y][x] /= gaussian_sum;
        }
    }

    // Wykonywanie filtra Gaussa na obrazie
    float3 color = (float3)(0.0f, 0.0f, 0.0f);

    for (int dy = -kernel_half_y; dy <= kernel_half_y; dy++) {
        for (int dx = -kernel_half_x; dx <= kernel_half_x; dx++) {
            int2 offset = pos + (int2)(dx, dy);
            float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, offset));
            
            float weight = gaussian_kernel[dy + kernel_half_y][dx + kernel_half_x];
            color += pixel.xyz * weight;
        }
    }

    // Zapisanie obliczonego koloru do obrazu wyjściowego
    write_imagef(outputImage, pos, (float4)(color, 1.0f));
}
