const int kernel_size = 20;
const int kernel_half = kernel_size/2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage){
    int2 pose = (int2)(get_global_id(0), get_global_id(1));
    float threshold = 120.0f;
    int min_value = 255.0f;

    for (int dx = -kernel_half; dx <= kernel_half; dx++){
        for (int dy = -kernel_half; dy <= kernel_half; dy++){
            if (dx*dx + dy*dy <= kernel_half*kernel_half){
                int2 dpose = pose + (int2)(dx, dy);
                uint3 pixel = read_imageui(inputImage, imageSampler, dpose).xyz;

                float3 pixel_float = (float3)(pixel.x, pixel.y, pixel.z);

                //printf("%f", pixel_float);
                float gray_pixel = dot(pixel_float, (float3)(0.299f, 0.587f, 0.114f));

                min_value = min(min_value, (int)(255*step(threshold, gray_pixel)*255));
            }     
        }
    }
    write_imagei(outputImage, pose, min_value);
}