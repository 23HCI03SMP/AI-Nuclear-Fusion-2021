#include <iostream>
#include <fftw3.h>
#include <math.h>
using namespace std;

int main()
{
    const size_t N=8;
    const size_t N0 = N+1, N1 =N +1, N2 = N+1;
    fftwf_plan planfor,planfor_k,planbac;
    float *arr,*arr1,*arr2,*arr3;
    arr = fftwf_alloc_real(N0*N1*N2);  //eg. particle charge distributed in the cells (then array replaced with "convolution kernel" e.g. Electric/potential  field of 1 point charge)
    arr1 = fftwf_alloc_real(N0*N1*N2); //array for fft of particle charge distribution then replaced with fft of partcle * fft of convolution kernel
    arr2 = fftwf_alloc_real(N0*N1*N2); //array for final result.
    arr3 = fftwf_alloc_real(N0*N1*N2);//array for fft of "convolution kernel"
    planfor = fftwf_plan_r2r_3d(N0,N1,N2, arr,arr1,FFTW_REDFT00,FFTW_REDFT00,FFTW_REDFT00, FFTW_ESTIMATE);
    planfor_k = fftwf_plan_r2r_3d(N0,N1,N2, arr,arr3,FFTW_REDFT00,FFTW_REDFT00,FFTW_REDFT00, FFTW_ESTIMATE);
    planbac = fftwf_plan_r2r_3d(N0,N1,N2, arr1,arr2,FFTW_REDFT00,FFTW_REDFT00,FFTW_REDFT00,FFTW_ESTIMATE);
    /* Allocate host & initialize data. */
    /* Only allocation shown for simplicity. */
    /* print input array just using the
     * indices to fill the array with data */
    printf("\nPerforming fft on an three dimensional array of size N0 x N1 x N2 : %lu x %lu x %lu\n", (unsigned long)N0, (unsigned long)N1, (unsigned long)N2);
    size_t i, j, k;
    i = j = k = 0;
    for (i=0; i<N0; ++i)
    {
        for (j=0; j<N1; ++j)
        {
            for (k=0; k<N2; ++k)
            {
                float x = 0.00f;
//             if (i==0 && j==0 && k==0)
                if (i==floor(N0/2) && j==floor(N1/2) && k==floor(N2/2))
                {
                    x = 1.0f;
                }
                size_t idx = (k+j*N2+i*N1*N2);
                arr[idx] = x;
                printf("%+1.1f, ", arr[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    fftwf_execute(planfor);// arr1 = fft(arr)
    /* print output array */
    printf("\n\nfft result: \n");
    i = j = k = 0;
    for (i=0; i<N0; ++i)
    {
        for (j=0; j<N1; ++j)
        {
            for (k=0; k<N2; ++k)
            {
                size_t idx = (k+j*N2+i*N1*N2);
                printf("%+1.1f, ", arr1[idx]/sqrt(8*(N0-1)*(N1-1)*(N2-1)));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    printf("\n\nkernel array: \n");
    for (i=0; i<N0; ++i)
    {
        float i2=(float)(i);//-N0*.5;
        i2*=i2;
        for (j=0; j<N1; ++j)
        {
            float j2=(float)(j);//-N1*0.5;
            j2*=j2;
            for (k=0; k<N2; ++k)
            {
                float k2=(float)(k);//-N2*0.5;
                k2*=k2;
                float x;
                float r=sqrt(i2+j2+k2);
                if (r<N0/2)   x = 1.0f/(r*r*r);
                else x =0.0f;


                if (i==0 && j==0 && k==0)
                {
                    x = 0.0f;
                }
                size_t idx = (k+j*N2+i*N1*N2);
                arr[idx] = x;
                printf("%1.1f, ", arr[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }



    fftwf_execute(planfor_k); //arr3= fft(arr)

    /* print output array */
    printf("\n\nfft of kernel: \n");
    i = j = k = 0;
    for (i=0; i<N0; ++i)
    {
        for (j=0; j<N1; ++j)
        {
            for (k=0; k<N2; ++k)
            {
                size_t idx = (k+j*N2+i*N1*N2);
                printf("%+1.1f, ", arr3[idx]/sqrt(8*(N0-1)*(N1-1)*(N2-1)));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    /* multiply fft charge with fft of kernel(i.e field associated with 1 charge */
    for (i=0; i<N0*N1*N2; ++i)
    {
        arr1[i]*=arr3[i];
    }

    /* inverse transform to get convolution */
    fftwf_execute(planbac);

    /* print output array */
    printf("\n\nfft inverse result: \n");
    i = j = k = 0;
    for (i=0; i<N0; ++i)
    {
        for (j=0; j<N1; ++j)
        {
            for (k=0; k<N2; ++k)
            {
                size_t idx = (k+j*N2+i*N1*N2);
                if  (arr2[idx]<10) arr2[idx]=0;
                printf("%+1.1f, ", arr2[idx]/(8*(N0-1)*(N1-1)*(N2-1)));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
    fftwf_destroy_plan(planfor);
    fftwf_destroy_plan(planbac);
    fftwf_destroy_plan(planfor_k);
    fftwf_free(arr);
    fftwf_free(arr1);
    fftwf_free(arr2);
    fftwf_free(arr3);
}
