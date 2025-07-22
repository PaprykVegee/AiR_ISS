const uint kernel_size = 5;
const int halfSize = kernel_size / 2;

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float threshold = 100.0f;
    bool dilatation_result = 0;


    for (int i = -halfSize; i <= halfSize; i++){
        for (int j = -halfSize; j <= halfSize; j++){
            int2 dcoord = coord + (int2)(i, j);

            // Sprawdzenie, czy punkt jest w obrębie koła
            if (i * i + j * j <= halfSize * halfSize) {
                float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, dcoord));

                float3 rgb = convert_float3(pixel.xyz);

                float gray_scale = dot(rgb, (float3)(0.2989f, 0.5870f, 0.1140f));

                bool binaryPixel = step(threshold, gray_scale);

                dilatation_result |= binaryPixel;
            }
        }
    }


    write_imageui(outputImage, coord, 255 * (uint)dilatation_result);
}