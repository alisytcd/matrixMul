#include <emmintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <pmmintrin.h>
#include <omp.h>

#pragma intrinsic( _mm_hadd_ps )

struct complex {
	float real;
	float imag;
};
// Copy the matrices to these matrices
struct complex ** A;
struct complex ** B;

struct complex Dot(__m128 *sseArrayA,__m128 *sseArrayB,int b_cols,int a_rows,int a_cols,int b_row){
	struct complex tmp,aC,bC;
	float ar,br,ai,bi;
	__m128 a, b,c, d,axb,cxd,bxa,dxc,subReal,addImg,resComp;
	int k,i,j;
	// iterate over a_rows
	for (j=0;j<a_rows;j++){
	// iterate over b_columns
	for (i=0;i<b_cols;i++){
		//iterate over col and rows
		for(k=0;k<(a_cols-4);k+4){
		// load from sseArrays the 4 values to be calculated so that each sse vector:
		// a = [complex real,complex img , complex real 1, complex img 1]
		 a = sseArrayA[k]; c = sseArrayA[k+1];
		 b = sseArrayB[k]; d = sseArrayB[k+1];
		 //  _mm_store_ss(&tmp, a);   
		 // printf("%d , %d \n",tmp.real,tmp.imag);
		//multiply a and b to get their real part
		 axb = _mm_mul_ps(a, b);
		//multiply c and d to get their real part
		 cxd = _mm_mul_ps(c, d);
		 //link for shuffle allocation: http://forums.codeguru.com/printthread.php?t=337156
		 // switch contents of b and d so that : [complex img, complex real, complex img 1, complex real 1]
		 b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));  
		 d = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)); 

		 //Multiply b and a , d and c to get their imaginary part
		 bxa = _mm_mul_ps(a, b);
		 dxc = _mm_mul_ps(c, d);
		 // Subtract the real parts since i^2 = -1 
		 subReal = _mm_hsub_ps(axb, cxd); 
		 // Add the imaginary parts
    	 addImg = _mm_hadd_ps(bxa,dxc);
    	 // Add real and imaginary to form Complex number
		 resComp = _mm_hadd_ps(subReal, addImg); 
    	 resComp = _mm_hadd_ps(resComp, resComp);  
    	 // Restore to float and Store the result complex number in a temp struct sum
    	  _mm_storel_pi((__m64 *) &tmp, resComp);
    	    sum.real += tmp.real;
    	    sum.imag += tmp.imag;
      	}
      	// If a_rows%2 != 0
    	for ( k; (k < k_max); k++) {
        	aC = A[i][k];
        	bC = B[k][i];
        	ar = aC.real; ai = aC.imag;
        	br = bC.real; bi = bC.imag;
        	sum.real += (ar * br) - (ai * bi);
        	sum.imag += (ar * bi) + (ai * br);
      	}
  		}
	}
	return sum;
 }
