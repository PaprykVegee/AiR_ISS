__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 


__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

    uint4 pixel = read_imageui(inputImage, imageSampler, coord);

    float3 rgb = convert_float3(pixel.xyz);

    float grayscaleValue = dot(rgb, (float3)(0.2989f, 0.5870f, 0.1140f));

    write_imageui(outputImage, coord, convert_uint4(grayscaleValue));	
}