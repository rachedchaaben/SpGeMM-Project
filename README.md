# SpGeMM-Project
This project consist of an SpGeMM implementation using CSR format, built based on Ginkgo. Both reference and OpenMP implemenation are available.
In this project, there is both iterative and recursive implementation of the same algorithm. 

##Testing and Benchmarking
We use in this code Google Test for testing the correctness of different components.
Also we use Google Benchmarks for the evaluation of the new SpGeMM.
##Compilation 
To compile and run this project, you need to have Ginkgo installed on your system(https://github.com/ginkgo-project/ginkgo), and also Google benchmarks (https://github.com/google/benchmark). OpenMP is required to be installed to compile the OMP implementation. 

Clone this repository :

git clone git@github.com:Rached-Chaaben/SpGeMM-Project.git
mkdir build && cd build 
cmake -DCMAKE_BUILD_TYPE=Release.. && make

Then you will have executables : spgemm and omp_spgemm 
