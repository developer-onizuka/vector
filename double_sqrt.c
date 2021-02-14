#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KB(x) ((x)*1024L)
#define N 8

void vector_sqrt(double *s, double *t, double *u, int n) {
	for(int i=0;i<n;i++) {
		u[i] = sqrt(s[i]*s[i] + t[i]*t[i]);
	}
}

int main(int argc, char *argv[])
{
	FILE *fpa,*fpb,*fpc,*fpd;
	double *a,*b,*c,*d,*x;
	int n;
	if(argc < 2) {
		n = N;
	} else {
		n = atoi(argv[1]);
	}
	a = (double*)malloc(sizeof(double)*n);
	b = (double*)malloc(sizeof(double)*n);
	for(int i=0;i<n;++i) {
		a[i] = 3.0;
		b[i] = 4.0;
	}

	fpa = fopen("./double_a.bin", "wr");
	fpb = fopen("./double_b.bin", "wr");
	fwrite(a, sizeof(double), n, fpa);
	fwrite(b, sizeof(double), n, fpb);
	fclose(fpa);
	fclose(fpb);

	c = (double*)malloc(sizeof(double)*n);
	d = (double*)malloc(sizeof(double)*n);
	x = (double*)malloc(sizeof(double)*n);
	fpc = fopen("./double_a.bin", "r");
	fpd = fopen("./double_b.bin", "r");
	fread(c, sizeof(double), n, fpc);
	fread(d, sizeof(double), n, fpd);
	vector_sqrt(c,d,x,n);
	for(int i=0;i<n;++i) {
		printf("output: %8.3lf\n", x[i]);
	}

	free(a);
	free(b);
	free(c);
	free(d);
	free(x);

	fclose(fpc);
	fclose(fpd);
}
