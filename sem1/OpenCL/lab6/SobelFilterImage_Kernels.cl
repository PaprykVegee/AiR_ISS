#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 3
#define KERNEL_STD 1.0f

float gaussian_kernel[KERNEL_HEIGHT][KERNEL_WIDTH];
float gaussian_sum = 0.0f;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

float gaussian_function(int x, int y, float std) {
    return exp(-((x * x + y * y) / (2.0f * std * std))) / (2.0f * M_PI * std * std);
}

void compute_gaussian_kernel() {
    int kernel_half_x = KERNEL_WIDTH / 2;
    int kernel_half_y = KERNEL_HEIGHT / 2;
    gaussian_sum = 0.0f;
    
    for (int y = -kernel_half_y; y <= kernel_half_y; y++) {
        for (int x = -kernel_half_x; x <= kernel_half_x; x++) {
            gaussian_kernel[y + kernel_half_y][x + kernel_half_x] = gaussian_function(x, y, KERNEL_STD);
            gaussian_sum += gaussian_kernel[y + kernel_half_y][x + kernel_half_x];
        }
    }
}

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    
    float color = 0.0f;  // Przechowujemy tylko jedną wartość dla obrazu czarno-białego
    
    int kernel_half_x = KERNEL_WIDTH / 2;
    int kernel_half_y = KERNEL_HEIGHT / 2;
    
    if (gaussian_sum == 0.0f) {
        compute_gaussian_kernel();
    }

    for (int dy = -kernel_half_y; dy <= kernel_half_y; dy++) {
        for (int dx = -kernel_half_x; dx <= kernel_half_x; dx++) {
            int2 offset = pos + (int2)(dx, dy);
            float4 pixel = read_imagef(inputImage, imageSampler, offset);
            
            float weight = gaussian_kernel[dy + kernel_half_y][dx + kernel_half_x];
            color += pixel.x * weight;  // Używamy tylko wartości 'x', ponieważ obraz jest czarno-biały
        }
    }
    
    color /= gaussian_sum;
    color = clamp(color, 0.0f, 1.0f);  // Upewniamy się, że wynik jest w zakresie [0, 1]
    
    // Zapisujemy wynik jako obraz czarno-biały (wszystkie składowe są takie same)
    write_imagef(outputImage, pos, (float4)(color, color, color, 1.0f));
}
