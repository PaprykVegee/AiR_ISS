const int kernel_size = 5;
const int kernel_half = kernel_size/2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    int2 pose = (int2)(get_global_id(0), get_global_id(1));

    float tabel[kernel_size * kernel_size];

    int i = 0;

    for (int dx = -kernel_half; dx <= kernel_half; dx++) {
        for (int dy = -kernel_half; dy <= kernel_half; dy++) {
            float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, pose + (int2)(dx, dy)));
            float3 rgb = pixel.xyz;
            float gray = dot(rgb, (float3)(0.2989f, 0.5870f, 0.1140f));
            tabel[i++] = gray;
        }
    }

    // Sortowanie bąbelkowe 
    int n = kernel_size * kernel_size;
    float temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (tabel[j] > tabel[j + 1]) {
                temp = tabel[j];
                tabel[j] = tabel[j + 1];
                tabel[j + 1] = temp;
            }
        }
    }

    // Obliczanie mediany
    int media_value;
    int middle = n / 2;

    if (n % 2 == 0) {
        media_value = (tabel[middle - 1] + tabel[middle]) / 2.0f;
    } else {
        media_value = tabel[middle];
    }

    write_imageui(outputImage, pose, media_value);
}