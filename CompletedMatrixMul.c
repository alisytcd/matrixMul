/*
	void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
		//replace this
		matmul(A, B, C, a_rows, a_cols, b_cols);
	}
*/

#include <emmintrin.h>
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

/* write matrix to stdout */
void write_out(struct complex ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("%.3f + %.3fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%.3f + %.3fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
  }
}

//TODO
//This function will have to be changed
/*
void complex_mul( struct complex a, struct complex b , __m128* sseArray, float* floatArray, result_row, result_column)
{

	sseArray[0] = _mm_set_ps(a.real, a.imag, a.real, a.imag);
	sseArray[1] = _mm_set_ps(b.real, -b.imag, b.imag, b.real);
	sseArray[2] = _mm_mul_ps(sseArray[0], sseArray[1]);

	sseArray[2] = _mm_hadd_ps(sseArray[2], sseArray[2]); 

	struct complex result;
	result.real = floatArray[9];
	result.imag = floatArray[8];
	return result;

}
*/
/*
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
*/


//complexMul(sseA, sseB, i * a_cols, j * a_cols, a_cols);

struct complex multest(__m128 * sseA, __m128 * sseB, int indexA, int indexB, int length) 
{
	__m128 sum = _mm_set_ps(0,0,0,0);
	for(int n = 0; n < length; n++){
		__m128 temp;
		temp = _mm_mul_ps( sseA[indexA + n], sseB[indexB + n] );
	
		#pragma omp critical
		{
		sum = _mm_add_ps(sum, temp);
		}
	}
	
	sum = _mm_hadd_ps(sum, sum);
	struct complex tmp;
	_mm_storel_pi((__m64*) &tmp, sum);
	printf("tmp.real = %f  \n", tmp.real);
	printf("tmp.imag = %f  \n", tmp.imag);
	float temporaryFloat = tmp.real;
	tmp.real =  tmp.imag;
	tmp.imag = temporaryFloat;
	return tmp;

}


void matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols)
{
	/*
		 Allocate Memory for sseArray
		------------------------------
		We'll start by allocating the memory we need.
		We will need 4 floats per complex number
		in order to multiply in one instruction.

		We will allocate space for an SSE Array representing both A and B
	*/

 	//We need the number of elements in each Matrix to decide how much space we need in our sseArray
 	int sizeof_A = a_rows * a_cols;
	int sizeof_B = a_cols * b_cols;

	//We need 2*2 floats for each complex number (to multiply in parallel)
	int ELEMENTS_NEEDED_A =  4 *  sizeof_A;
	int ELEMENTS_NEEDED_B =  4 *  sizeof_B;
	
	//We declare the float array, then align it to 16 bytes, then cast it to __m128
	float *floatsA;
  	posix_memalign((void**)&floatsA, 16,  ELEMENTS_NEEDED_A * sizeof(float));
 	__m128 *sseA = (__m128*) floatsA;
 	//Both arrays (float and __m128) point to the same data!

 	float *floatsB;
  	posix_memalign((void**)&floatsB, 16,  ELEMENTS_NEEDED_B * sizeof(float));
 	__m128 *sseB = (__m128*) floatsB;
 	
 	


 	/*
		 Initialise sseArray
		---------------------
 		We could take this opportunity to populate our sseArray with values from A and B
 		It could go like this:
		
			//one thread
			int index = 0;
			for(i = 0; i < a_rows; i++)
				for(j = 0; j < a_cols; j++)
					sseArray [index] = _mm_set_ps( A[i][j].real, A[i][j].imag, A[i][j].real, A[i][j].imag );
	
			//another thread
			int index = #[A] //the number of elements in A so we don't overlap
			for(i = 0; i < a_rows; i++)
				for(j = 0; j < b_cols; j++)
					sseArray [index] = _mm_set_ps( B[i][j].real, - B[i][j].imag, B[i][j].imag, B[i][j].real );
 	*/
	int indexA = 0;
	for (int i = 0; i < a_rows; i++){
		for (int j = 0; j < a_cols; j++){
			sseA[indexA] = _mm_set_ps( A[i][j].real, A[i][j].imag, A[i][j].real, A[i][j].imag );
			indexA++;
		}
	}

	int indexB = 0;
	for (int j = 0; j < b_cols; j++){
		for (int i = 0; i < a_cols; i++){
			sseB[indexB] = _mm_set_ps( B[i][j].real, -B[i][j].imag, B[i][j].imag, B[i][j].real );
			//printf("Setting sseB %d to %f %f %f %f\n", indexB, B[i][j].real, -B[i][j].imag, B[i][j].imag, B[i][j].real );
			//printf("Setting sseB %d from %d %d \n", indexB, i, j );
			indexB++;
		}
	}

	/*
	for (int i = 0; i < ELEMENTS_NEEDED_B; i++){
		printf("floatsB %d:  %f \n", i, floatsB[i]);
	}
	*/

	
	struct complex sumProds;
	#pragma omp for 
	for(int i=0;i<a_rows;i++){

		for(int j=0;j<b_cols;j++){
    		C[i][j] = multest(sseA, sseB, i * a_cols, j * a_cols, a_cols);

		}
	}	
}

/* create new empty matrix */
struct complex ** new_empty_matrix(int dim1, int dim2)
{
  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);
  int i;

  for ( i = 0; i < dim1; i++ ) {
    result[i] = &(new_matrix[i*dim2]);
  }

  return result;
}



	/*
	*	A:	[	00		10	] 		B:	[	00		10	]
	*		[ 	01		11	]			[	01		11	]
	*
	*
	*	sseA		A[0][0]		A[1][0]		A[0][1]		A[1][1]
	*
	*	sseB		B[0][0]		B[0][1]		B[1][0]		B[1][1]
	*
	*/


int main(){

	int dim = 2;	

	struct complex ** A = new_empty_matrix(dim, dim);
	int counter = 0;
	for (int i = 0; i < dim; i++){
		for (int j = 0; j< dim; j++){
			A[i][j].real = counter;
			A[i][j].imag = counter;
			//printf("Setting B %d %d = to   %d   %d\n", i, j, counter, counter);
			counter++;
		}

	}
	
	struct complex ** B = new_empty_matrix(dim, dim);
	counter = 0;
	for (int i = 0; i < dim; i++){
		for (int j = 0; j< dim; j++){
			B[i][j].real = counter;
			B[i][j].imag = counter;
			//printf("Setting B %d %d = to   %d   %d\n", i, j, counter, counter);
			counter++;
		}

	}
	struct complex ** C = new_empty_matrix(dim, dim);

	matmul(A, B, C, dim, dim, dim);
	printf("A:\n");
	write_out(A, dim, dim);
	printf("B:\n");
	write_out(B, dim, dim);
	printf("C:\n");
	write_out(C, dim, dim);
}

