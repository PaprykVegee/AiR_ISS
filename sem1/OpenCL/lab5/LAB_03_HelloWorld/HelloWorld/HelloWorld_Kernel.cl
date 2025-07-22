__kernel void helloworld(__global char* in, __global char* out)
{
    int num = get_global_id(0);

    int decimal_value = 0;

    int hex = in[num];

    // Przekształcanie znaków heksadecymalnych na wartości dziesiętne
    if (hex >= 'a' && hex <= 'f'){
        decimal_value = hex - 'a' + 10;
    } else if (hex >= 'A' && hex <= 'F'){
        decimal_value = hex - 'A' + 10;
    } else if (hex >= '0' && hex <= '9'){
        decimal_value = hex - '0';
    } else {
        decimal_value = 0; // nieobsługiwany znak
    }

	// Obliczanie bitów
	for (int i = 7; i >= 0; i--) {
		// Wydzielanie pojedynczego bitu
		int bit = (decimal_value >> i) & 1;

		// Zapisz bit w odpowiedniej pozycji w tablicy wyjściowej
		out[8 * num + (7 - i)] = (char)(bit + '0');  // zamień na znak '0' lub '1'
	}
}
