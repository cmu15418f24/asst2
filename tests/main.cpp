#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

//void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
//void printCudaInfo();


// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}


int main(int argc, char** argv)
{
    printf("ENTERING MAIN.CPP MAIN");\
//    saxpyCuda(N, alpha, xarray, yarray, resultarray);

//    printCudaInfo(); # TODO try running printcudainfo
//
//
//    delete [] xarray;
//    delete [] yarray;
//    delete [] resultarray;

    return 0;
}
