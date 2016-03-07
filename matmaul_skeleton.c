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

//TODO
//This function will have to be changed
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


struct complex** matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols)
{
	/*
		 Allocate Memory for sseArray
		------------------------------
		We'll start by allocating the memory we need.
		We will need 4 floats per complex number
		in order to multiply in one instruction.

	*/

 	//We need the number of elements in each Matrix to decide how much space we need in our sseArray
 	int sizeof_A = a_rows * a_cols;
	int sizeof_B = a_cols * b_cols;
	int sizeof_C = a_rows * b_cols;

	//We need 2*2 floats for each complex number (to multiply in parallel)
	int ELEMENTS_NEEDED =  4 * ( sizeof_A + sizeof_B + sizeof_C );
	
	//We declare the float array, then align it to 16 bytes, then cast it to __m128
	float *floatArray;
  	posix_memalign((void**)&floatArray, 16,  ELEMENTS_NEEDED * sizeof(float));
 	__m128 *sseArray = (__m128*) floatArray;
 	
 	//Both arrays point to the same data!


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

	/*		
		
 		For now we'll check the dimensions of the matrices
 		so that we can pick the best for loop structure
 		in order to maximise cache hits.

 	
 			if ( a_rows > b_cols )
				for i < b_cols...
		 			for j < a_rows...
			else
				//do it the other way round

	*/

	if ( a_rows > b_cols )
	{
		
		// We will iterate through B's columns on the outer loop

		#pragma omp parallel for
 		for (int i = 0; i < b_cols; i++)
 			
 			#pragma omp parallel for
 			for (int j = 0; j < a_rows; j++)
 				struct complex sum;
 				sum.real = 0.0;
 				sum.imag = 0.0;
				
				#pragma omp parallel for
 				for (int k = 0; k < a_cols; k++)
 					/*
						 Multiply
						----------
	 					Here's where it gets hazy
	 					We could make a struct complex product here
	 					and then add the real and imaginary parts to that complex sum we declared
	 					straight forward but probably slow

	 					or we could use the space we allocated for C in the sseArray above
	 					and store all the products there in the form:  sseArray[x] == (x.real, x.real, x.imag, x.imag)
	 					then do one massive parallel-adding for loop and put each result straight into C[j][i] in parallel
					*/

	}


 	/*

		


		 Notes
		-------
		- It might be beneficial to allocate space for C (instead of overwriting A or B)
		since the elements of A and B will be reused.
		This depends on: 
		the speed of allocating space for C (and the available space, which we might not have for huge matrices)
			vs
		the speed of writing to the array with _mm_set_ps(b.real, -b.imag, b.imag, b.real)
		If we _mm_set_ps all of A and B beforehand,
		I think it will be faster to allocate space for C.


		- To minimize cache misses, we can check which is larger: a_rows vs b_cols

			 [A]			 [B]
		0	0	0	0		0	0		[B]'s columns will be accessed more often (a_rows times)
		0	0	0	0		0	0		where [A]'s rows are only accessed b_cols times 
		0	0	0	0		0	0		[B] col accesses:	4
		0	0	0	0		0	0		[A] col accesses:	2


		0	0	0	0		0	0	0	0	[A] row accesses: 	4
		0	0	0	0		0	0	0	0	[B] col accesses:	2
							0	0	0	0
							0	0	0	0
		
		Basically, we should construct the for loops so that the larger of a_rows and b_cols is on the inside
		so that the cache will keep the row/column that is used most often.
		This way is a little naive though;
		  if both a_rows and b_cols are too big to fit in the cache  then we might want to do something else
		      like go forwards through A[0 - N] x B[0]
		      then backwards through A[N - 0] x B[1]



		- if we can get all the products into one array, 
		then we could use the horizontal add more efficiently
	*/

}
