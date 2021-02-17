#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define KB(x) ((x)*1024L)
#define N 8 

__global__ void vector_sqrt(float *s, float *t, float *u) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	u[idx] = sqrt(s[idx]*s[idx] + t[idx]*t[idx]);
	/* printf("idx:%d, blockId.x:%d, threadIdx.x:%d\n",idx, blockIdx.x, threadIdx.x); */
}

int main(int argc, char *argv[])
{
	FILE *fpa,*fpb,*fpc,*fpd;
	float *a,*b,*c,*d,*x;
	float *c_d,*d_d,*x_d;
	int n;
	if(argc < 2) {
		n = N;
	} else {
		n = atoi(argv[1]);
	}
	a = (float*)malloc(sizeof(float)*n);
	b = (float*)malloc(sizeof(float)*n);
	for(int i=0;i<n;++i) {
		a[i] = 3.0;
		b[i] = 4.0;
	}

	fpa = fopen("./float_a.bin", "wr");
	fpb = fopen("./float_b.bin", "wr");
	fwrite(a, sizeof(float), n, fpa);
	fwrite(b, sizeof(float), n, fpb);
	fclose(fpa);
	fclose(fpb);

	c = (float*)malloc(sizeof(float)*n);
	d = (float*)malloc(sizeof(float)*n);
	x = (float*)malloc(sizeof(float)*n);
	cudaMalloc(&c_d, sizeof(float)*n);
	cudaMalloc(&d_d, sizeof(float)*n);
	cudaMalloc(&x_d, sizeof(float)*n);
	fpc = fopen("./float_a.bin", "r");
	fpd = fopen("./float_b.bin", "r");
	fread(c, sizeof(float), n, fpc);
	fread(d, sizeof(float), n, fpd);
	cudaMemcpy(c_d, c, sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d, sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x, sizeof(float)*n, cudaMemcpyHostToDevice);

	int blocksize = 512;
	int gridsize = (n+(blocksize-1))/blocksize;
	dim3 dimGrid(gridsize,1);
	dim3 dimBlock(blocksize,1,1);	
	vector_sqrt<<<dimGrid,dimBlock>>>(c_d,d_d,x_d);

	cudaMemcpy(x, x_d, sizeof(float)*n, cudaMemcpyDeviceToHost);
	for(int i=0;i<n;++i) {
		printf("output: %5.3e\n", x[i]);
	}
	/* printf("dimGrid:%d, dimBlock:%d\n", gridsize, blocksize); */

	free(a);
	free(b);
	free(c);
	free(d);
	free(x);
	cudaFree(c_d);
	cudaFree(d_d);
	cudaFree(x_d);

	fclose(fpc);
	fclose(fpd);
}
