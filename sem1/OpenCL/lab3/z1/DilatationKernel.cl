
/*
const uint kernel_size = 20;
const int half_kernel = kernel_size / 2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float max_r = 0.0f;
    float max_g = 0.0f;
    float max_b = 0.0f;

    for (int i = -half_kernel; i <= half_kernel; i++) {
        for (int j = -half_kernel; j <= half_kernel; j++) {
            int2 neighborPos = pos + (int2)(i, j);
            float4 pixel_val = read_imagef(inputImage, imageSampler, neighborPos);

            max_r = max(max_r, pixel_val.x);
            max_g = max(max_g, pixel_val.y);
            max_b = max(max_b, pixel_val.z);
        }
    }

    write_imagef(outputImage, pos, (float4)(max_r, max_g, max_b, 1.0f));
}
*/


const uint kernel_size = 20;
const int halfSize = kernel_size / 2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float threshold = 100.0f;
    bool dilatation_result = 0;

    for (int i = -halfSize; i <= halfSize; i++){
        for (int j = -halfSize; j <= halfSize; j++){
            float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, coord + (int2)(i, j)));

            float3 rgb = convert_float3(pixel.xyz);

            float gray_scale = dot(rgb, (float3)(0.2989f, 0.5870f, 0.1140f));

            bool binaryPixel = step(threshold, gray_scale);

            dilatation_result |= binaryPixel;
        }
    }

    write_imageui(outputImage, coord, 255 * (uint)dilatation_result);
}
