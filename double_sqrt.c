#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KB(x) ((x)*1024L)
#define N 8 

void main(int argc, char *argv[])
{
	FILE *fpa,*fpb,*fpc,*fpd;
	double *a,*b,*c,*d,*x;
	a = (double*)malloc(sizeof(double)*N);
	b = (double*)malloc(sizeof(double)*N);
	for(int i=0;i<N;++i) {
		a[i] = 3.0;
		b[i] = 4.0;
	}

	fpa = fopen("./double_a.txt", "wr");
	fpb = fopen("./double_b.txt", "wr");
	fwrite(a, sizeof(double), N, fpa);
	fwrite(b, sizeof(double), N, fpb);
	fclose(fpa);
	fclose(fpb);

	c = (double*)malloc(sizeof(double)*N);
	d = (double*)malloc(sizeof(double)*N);
	x = (double*)malloc(sizeof(double)*N);
	fpc = fopen("./double_a.txt", "r");
	fpd = fopen("./double_b.txt", "r");
	fread(c, sizeof(double), N, fpc);
	fread(d, sizeof(double), N, fpd);
	for(int j=0;j<N;++j) {
		x[j] = sqrt(c[j]*c[j] + d[j]*d[j]);
		printf("c: %8.3lf,    d: %8.3lf   ", c[j], d[j]);
		printf("x: %8.3lf\n", x[j]);
	}

	free(a);
	free(b);
	free(c);
	free(d);
	free(x);

	fclose(fpa);
	fclose(fpb);
}
