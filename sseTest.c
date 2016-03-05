/*
    SSE Testing
    Compile this with:
      gcc -std=c99 -msse3 sseTest.c
*/


#include <emmintrin.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <pmmintrin.h>

#pragma intrinsic( _mm_hadd_ps )

struct complex {
  float real;
  float imag;
};   

struct complex complex_mul( struct complex a, struct complex b , __m128* sseArray, float* floatArray)
{

  sseArray[0] = _mm_set_ps(a.real, a.imag, a.real, a.imag);
  sseArray[1] = _mm_set_ps(b.real, -b.imag, b.imag, b.real);
  sseArray[1] = _mm_mul_ps(sseArray[0], sseArray[1]);

  sseArray[1] = _mm_hadd_ps(sseArray[1], sseArray[1]);

  struct complex result;
  result.real = floatArray[5];
  result.imag = floatArray[4];
  return result;

}

void test_complex_mul()
{
  
  float *floatArray;
  posix_memalign((void**)&floatArray, 16,  8 * sizeof(float));
  __m128 *sseArray = (__m128*) floatArray;

  struct complex X, Y, Z;
  X.real = -2;
  X.imag = 2;
  Y.real = 5;
  Y.imag = 3;

  Z = complex_mul(X, Y, sseArray, floatArray);

  printf("real: %5.3f\ncomplex: %5.3f\n", Z.real, Z.imag);

  printf("floatArray:\n");
  for (int i = 0; i < 12; i++){
    printf("%d: %5.3f \n", i, floatArray[i]);

  }

}

 
int main()
{
  test_complex_mul();  
}
