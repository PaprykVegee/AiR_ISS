
/*
const uint kernel_size = 15;
const int half_kernel = kernel_size / 2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float min_r = 1.0f;
    float min_g = 1.0f;
    float min_b = 1.0f;

    for (int i = -half_kernel; i <= half_kernel; i++) {
        for (int j = -half_kernel; j <= half_kernel; j++) {
            int2 neighborPos = pos + (int2)(i, j);
            float4 pixel_val = read_imagef(inputImage, imageSampler, neighborPos);

            min_r = min(min_r, pixel_val.x);
            min_g = min(min_g, pixel_val.y);
            min_b = min(min_b, pixel_val.z);
        }
    }

    write_imagef(outputImage, pos, (float4)(min_r, min_g, min_b, 1.0f)); 
}
*/
const uint kernel_size = 20;
const int halfSize = kernel_size / 2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float threshold = 100.0;
    bool wynik_erozji = 1; 

    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, coord + (int2)(i, j)));
            
            float grayValue = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
            
            bool binaryPixel = step(threshold, grayValue);
            
            wynik_erozji &= binaryPixel;
        }
    }
    
    // Zapisujemy wynik erozji (0 lub 1) do obrazu wyjściowego jako biały (255) lub czarny (0)
    write_imageui(outputImage, coord, 255 * (uint)wynik_erozji);
}
