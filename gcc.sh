gcc double_sqrt.c -o double_sqrt.o -lm
nvcc double_sqrt.cu -o double_sqrt.co -lm
nvcc -I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ double_sqrt_gds.cu -o double_sqrt_gds.co -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L   -Bstatic -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl -lcrypto -lssl

