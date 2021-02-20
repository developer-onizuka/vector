#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define KB(x) ((x)*1024L)
#define N 8 

__global__ void vector_sqrt(double *s, double *t, double *u) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	u[idx] = sqrt(s[idx]*s[idx] + t[idx]*t[idx]);
	/* printf("idx:%d, blockId.x:%d, threadIdx.x:%d\n",idx, blockIdx.x, threadIdx.x); */
}

int main(int argc, char *argv[])
{
	FILE *fpa,*fpb,*fpc,*fpx,*fpy,*fpz;
	double *a,*b,*c;
	double *x,*y,*z;
	double *x_d,*y_d,*z_d;
	int n;
	if(argc < 2) {
		n = N;
	} else {
		n = atoi(argv[1]);
	}
	a = (double*)malloc(sizeof(double)*n);
	b = (double*)malloc(sizeof(double)*n);
	c = (double*)malloc(sizeof(double)*n);
	for(int i=0;i<n;++i) {
		a[i] = 3.0;
		b[i] = 4.0;
		c[i] = 0.0;
	}

	fpa = fopen("./double_a.bin", "wr");
	fpb = fopen("./double_b.bin", "wr");
	fpc = fopen("./double_c.bin", "wr");
	fwrite(a, sizeof(double), n, fpa);
	fwrite(b, sizeof(double), n, fpb);
	fwrite(c, sizeof(double), n, fpc);
	fclose(fpa);
	fclose(fpb);
	fclose(fpc);

	x = (double*)malloc(sizeof(double)*n);
	y = (double*)malloc(sizeof(double)*n);
	z = (double*)malloc(sizeof(double)*n);
	cudaMalloc(&x_d, sizeof(double)*n);
	cudaMalloc(&y_d, sizeof(double)*n);
	cudaMalloc(&z_d, sizeof(double)*n);

	fpx = fopen("./double_a.bin", "r");
	fpy = fopen("./double_b.bin", "r");
	fpz = fopen("./double_c.bin", "rw");

	fread(x, sizeof(double), n, fpx);
	fread(y, sizeof(double), n, fpy);
	fread(z, sizeof(double), n, fpz);

	cudaMemcpy(x_d, x, sizeof(double)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(z_d, z, sizeof(double)*n, cudaMemcpyHostToDevice);

	int blocksize = 512;
	int gridsize = (n+(blocksize-1))/blocksize;
	dim3 dimGrid(gridsize,1);
	dim3 dimBlock(blocksize,1,1);	
	vector_sqrt<<<dimGrid,dimBlock>>>(x_d,y_d,z_d);

	cudaMemcpy(z, z_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
	for(int i=0;i<n;++i) {
		printf("output: %8.3lf\n", z[i]);
	}
	/* printf("dimGrid:%d, dimBlock:%d\n", gridsize, blocksize); */

	free(a);
	free(b);
	free(c);
	free(x);
	free(y);
	free(z);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);

	fclose(fpa);
	fclose(fpb);
	fclose(fpc);
}
