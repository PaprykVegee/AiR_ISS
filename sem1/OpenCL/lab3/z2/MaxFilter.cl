const uint kernel_size = 30;
const int halfSize = kernel_size / 2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    int max_value = 0;

    int threshold = 120;
    
    for (int dy = -halfSize; dy <= halfSize; dy++) {
        for (int dx = -halfSize; dx <= halfSize; dx++) {
            int2 coord = pos + (int2)(dx, dy);
            uint3 pixel = read_imageui(inputImage, imageSampler, coord).xyz;

            float3 pixel_float = (float3)(pixel.x, pixel.y, pixel.z);
            
            float gray_value = dot(pixel_float, (float3)(0.299f, 0.587f, 0.114f));
            
            max_value = max(max_value, (int)(255 * step(threshold, gray_value)));
        }
    }
    
    write_imageui(outputImage, pos, max_value);
}
