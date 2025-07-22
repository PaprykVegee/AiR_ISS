__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

const float threshold = 0.2; 

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

    uint4 pixel = read_imageui(inputImage, imageSampler, coord);

    float3 noramlized = convert_float3(pixel.xyz) / 255.0f;

    noramlized = step(threshold, noramlized)*255;

    write_imageui(outputImage, coord, pixel);		
}