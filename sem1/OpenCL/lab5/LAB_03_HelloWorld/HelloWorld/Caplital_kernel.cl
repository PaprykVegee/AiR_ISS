__kernel void helloworld(__global char* in, __global char* out)
{
    int num = get_global_id(0);

    char current_char = in[num]; 

    if (current_char >= 'a' && current_char <= 'z') {
        out[num] = current_char - 'a' + 'A';
    } else if (current_char >= 'A' && current_char <= 'Z') {
        out[num] = current_char - 'A' + 'a';
    } else {
        out[num] = current_char;
    }
}