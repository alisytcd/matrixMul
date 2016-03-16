/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>

#include <pmmintrin.h>
#include <emmintrin.h>

#pragma intrinsic( _mm_hadd_ps )


/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

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

void free_matrix(struct complex ** matrix) {
  free (matrix[0]); /* free the contents */
  free (matrix); /* free the header */
}

/* take a copy of the matrix and return in a newly allocated matrix */
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
  int i, j;
  struct complex ** result = new_empty_matrix(dim1, dim2);

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
struct complex ** gen_random_matrix(int dim1, int dim2)
{
  const int random_range = 512; // constant power of 2
  struct complex ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      /* evenly generate values in the range [0, random_range-1)*/
      result[i][j].real = (float)(random() % random_range);
      result[i][j].imag = (float)(random() % random_range);

      /* at no loss of precision, negate the values sometimes */
      /* so the range is now (-(random_range-1), random_range-1)*/
      if (random() & 1) result[i][j].real = -result[i][j].real;
      if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
/* returns number of differing values */
void check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
{
  int i, j;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      double diff;
      diff = abs(control[i][j].real - result[i][j].real);
      sum_abs_diff = sum_abs_diff + diff;

      diff = abs(control[i][j].imag - result[i][j].imag);
      sum_abs_diff = sum_abs_diff + diff;
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
      sum_abs_diff, EPSILON);
  }
}

/* multiply matrix A times matrix B and put result in matrix C */
void matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2)
{
  int i, j, k;

  for ( i = 0; i < a_dim1; i++ ) {
    for( j = 0; j < b_dim2; j++ ) {
      struct complex sum;
      sum.real = 0.0;
      sum.imag = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
        // the following code does: sum += A[i][k] * B[k][j];
        struct complex product;
        product.real = A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
        product.imag = A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
        sum.real += product.real;
        sum.imag += product.imag;
      }
      C[i][j] = sum;
    }
  }
}

/* 
  ---------------------------------------------------------------------------------------------
  mul_row_column
  ---------------------------------------------------------------------------------------------
  Multiplies row by column and returns the result in a struct complex

  Input matrices are found in two SSE arrays: sseA, sseB

  The SSE arrays will be received alongside an index: indexA, indexB
  Specific complex numbers can be accessed by adding an offset (n) to the index
  where sseA[indexA + n] is the nth complex number in A's row
    and sseB[indexB + n] is the nth complex number in B's column (by virtue of how sseB is populated)

  Each complex number will be found in this form:
        sseA[indexA + n]   -->   |  A.r  |  A.i  |  A.r  |  A.i  |
        sseB[indexB + n]   -->   |  B.r  |(-)B.i |  B.i  |  B.r  |

  Note:
  1.  Each real and each imaginary are stored twice, in a prticular order.
  2.  The second element of sseB[indexB + n] is negated.
  These properties allow us to perform a single vector multiplication for each complex number
  Giving us:
        temp               -->   |  t.r1  |  t.r2  |  t.i1  |  t.i2  |

  We can just parallel add temp to our running sum each time until we have finished.
  Then we horizontal add our finished sum to combine the real parts and the imaginary parts.

        sum                -->   |  s.r  |  s.i  |  s.r  |  s.i  |

  We end up with some duplicate values at the end, 
  but we can ignore those and extract our complex number
  from the first two places in the final __m128 sum

  ---------------------------------------------------------------------------------------------
 */
struct complex mul_row_column(__m128 * sseA, __m128 * sseB, int indexA, int indexB, int length) 
{

  __m128 sum = _mm_set_ps(0,0,0,0);
  
  int n;
  for(n = 0; n < length; n++){
    __m128 temp;
    temp = _mm_mul_ps( sseA[indexA + n], sseB[indexB + n] );
    sum = _mm_add_ps(sum, temp);
  }

  //Combine real parts and imaginary parts
  sum = _mm_hadd_ps(sum, sum);

  //Extract complex number from __m128 sum
  struct complex result;
  _mm_storel_pi((__m64*) &result, sum);

  //real and imag parts are in the wrong place after _mm_storel_pi
  //So we need to swap them;
  float temporaryFloat = result.real;
  result.real =  result.imag;
  result.imag = temporaryFloat;
  return result;

}


void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols)
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
    We will take this opportunity to populate our SSE Arrays with values from A and B
    The following is where the SSE Array Indices will point
    
    A:
      [ 0   1   2   3   ]
      [ 4   5   6   7   ]
      [ 8   9   10  11  ]
      [ 12  13  14  15  ]
    B:
      [ 0   4   8   12  ]
      [ 1   5   9   13  ]
      [ 2   6   10  14  ]
      [ 3   7   11  15  ]

    B is allocted in this way so that it is easier to access its columns.
    Since we will be accessing each element in a column in sequence,
      it makes sense to store these elements in sequence.
    This might prevent some cache misses.
    
    Also, when we multiply a row and a column, we can say:
      for (n = 0; n < row_column_length; n++)
        product = A[n] * B[n]
  */
 
  
  //Set SSE array A		
  int indexA = 0;
  for (int i = 0; i < a_rows; i++){
    for (int j = 0; j < a_cols; j++){
      sseA[indexA] = _mm_set_ps( A[i][j].real, A[i][j].imag, A[i][j].real, A[i][j].imag );
      indexA++;
    }
  }
  
	//Set SSE array B
  int indexB = 0;
  for (int j = 0; j < b_cols; j++){
    for (int i = 0; i < a_cols; i++){
      sseB[indexB] = _mm_set_ps( B[i][j].real, -B[i][j].imag, B[i][j].imag, B[i][j].real );
		  indexB++;
    }
  }
  
  /*
         Multiply Matrices
      -----------------------
      Now that the matrices are loaded into SSE arrays,
      we can set about multiplying them. 
     
      Note:
      for all i in C_columns
         for all j in C_rows
            C[i][j] is independant of C[x][y] for all x != i and y != j
      
      Because of this independance, we can iterate through C using a parallel for.
      
      The catch is:
      for very narrow matrices, 
      the time spent setting up threads 
      is more than the time multithreaded parallelism can save.
      
      A 'threshold of worth' for each dimension was found
      that is used to decide whether to set up threads or not.
      These thresholds are specific to Stoker.
  */
  
  //The following are the minimum dimensions of Input Matrices where using 
  //  #pragma omp parallel for 
  //gives a speedup
  
  //outerMin -> Approximate minimum value for a_rows and b_cols
  int outerMin = 50;

  //innerMin -> Approximate minimum value for a_cols
  int innerMin = 5;  
  
  int i, j; //need to declare these outside omp for
  #pragma omp parallel for private(i, j) collapse(2) if(a_rows > outerMin && b_cols > outerMin && a_cols > innerMin)
  for(i=0;i<a_rows;i++){
    for(j=0;j<b_cols;j++){
        C[i][j] = mul_row_column(sseA, sseB, i * a_cols, j * a_cols, a_cols);
    }
  } 
}


long long time_diff(struct timeval * start, struct timeval * end) {
  return (end->tv_sec - start->tv_sec) * 1000000L + (end->tv_usec - start->tv_usec);
}

int main(int argc, char ** argv)
{
  struct complex ** A, ** B, ** C;
  struct complex ** control_matrix;
  long long control_time, mul_time;
  double speedup;
  int a_dim1, a_dim2, b_dim1, b_dim2, errs;
  struct timeval pre_time, start_time, stop_time;

  if ( argc != 5 ) {
    fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
    exit(1);
  }
  else {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  /* check the matrix sizes are compatible */
  if ( a_dim2 != b_dim1 ) {
    fprintf(stderr,
      "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
      a_dim2, b_dim1);
    exit(1);
  }

  /* allocate the matrices */
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  control_matrix = new_empty_matrix(a_dim1, b_dim2);

  DEBUGGING( {
    printf("matrix A:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\nmatrix B:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\n");
  } )

  /* record control start time */
  gettimeofday(&pre_time, NULL);

  /* use a simple matmul routine to produce control result */
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* perform matrix multiplication */
  team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);

  /* compute elapsed times and speedup factor */
  control_time = time_diff(&pre_time, &start_time);
  mul_time = time_diff(&start_time, &stop_time);
  speedup = (float) control_time / mul_time;

  printf("Matmul time: %lld microseconds\n", mul_time);
  printf("control time : %lld microseconds\n", control_time);
  if (mul_time > 0 && control_time > 0) {
    printf("speedup: %.2fx\n", speedup);
  }

  /* now check that the team's matmul routine gives the same answer
     as the known working version */
  check_result(C, control_matrix, a_dim1, b_dim2);

  /* free all matrices */
  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(control_matrix);

  return 0;
}
