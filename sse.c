#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

struct vector{
	float realv[4];
	float imv[4];
};

struct vector * new_empty_vector(){ 
	struct vector * new_vector = malloc(sizeof(struct vector));
 	int i;
	for(i=0;i<4;i++){
		new_vector->realv[i]=0;
 		new_vector->imv[i]=0;
	}
  
  return new_vector;
}

struct vector * vectormul(struct vector * v1,struct vector * v2){
	struct vector * product = new_empty_vector();
	int i;
	#pragma omp parallel for private(i)
		for (i=0;i<4;i++){
			product->realv[0]=(v1->realv[i]*v2->realv[i])-(v1->imv[i]*v2->imv[i]);
			product->imv[0]=(v1->realv[i]*v2->imv[i])+(v1->imv[i]*v2->realv[i]);
		}
	return product;
}

int main(int argc, char ** argv)
{  
	return 0;
}