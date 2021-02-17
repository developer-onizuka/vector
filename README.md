```
$ time ./double_sqrt.co 67108864 |tail -n 1
output:    5.000

real	1m13.499s
user	1m11.908s
sys	0m2.431s
$ time ./double_sqrt_gds.co 67108864 |tail -n 1
output:    5.000

real	0m14.627s
user	0m11.243s
sys	0m2.050s
$ time ./double_sqrt.o 67108864 |tail -n 1
output:    5.000

real	0m12.586s
user	0m10.689s
sys	0m2.158s
```
