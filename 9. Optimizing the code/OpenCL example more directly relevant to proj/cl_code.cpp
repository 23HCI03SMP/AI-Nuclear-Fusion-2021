#include "traj.h"

void cl_start(cl::Context &context1,cl::Device &default_device1,cl::Program &program1)
{
//get all platforms (drivers)
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::vector<cl::Device> devices;
    int platform_id = 0;
    int device_id = 0;


    std::cout << "Number of Platforms: " << platforms.size() << std::endl;

    for(cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
    {
        cl::Platform platform(*it);

        std::cout << "Platform ID: " << platform_id++ << std::endl;
        std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

        //     platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
//       platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        for(cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
        {
            cl::Device device(*it2);
            std::cout << "Number of Devices: " << devices.size() << std::endl;
            std::cout << "\tDevice " << device_id++ << ": " << std::endl;
            std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
            std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
            std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\t\tDevice Global Memory: MB " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1024/1024 << std::endl;
            std::cout << "\t\tDevice Max Clock Frequency: MHz " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            std::cout << "\t\tDevice Max Allocateable Memory MB: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/1024/1024 << std::endl;
            std::cout << "\t\tDevice Local Memory: kB " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024 << std::endl;
            std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
        }
        std::cout<< std::endl;
    }


    cl::Platform::get(&platforms);
    cl::Platform default_platform=platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device default_device=devices[0];
    std::cout << "\t\tDevice Name: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context({default_device});

    cl::Program::Sources sources;

    // kernel calculates for each element C=A+B
    std::string kernel_code=
        "   void kernel calc_pnn_k("
        "    global  float* px            , global  float* py            , global  float* pz, "
        "    global  float* qdt2_2mEx     , global  float* qdt2_2mEy     , global  float* qdt2_2mEz, "
        "    global float* pos0x          , global float* pos0y          , global float* pos0z, "
        "    global float* pos1x          , global float* pos1y          , global float* pos1z, "
        "    global float* pxx            , global float* pxy            , global float* pxz, "
        "    global float* pyy            , global float* pyz            , global float* pzz, "
        "    global float* dx             , global float* dy             , global float* dz, "
        "    global float* r_determinant, "
        "    global float* M1             , global float* M2             , global float* M3,"
        "    global float* co ){    "
        "       int id=get_global_id(0);           \n"
        "       qdt2_2mEx[id]=co[1]*qdt2_2mEx[id]; \n"
        "       qdt2_2mEy[id]=co[1]*qdt2_2mEy[id]; \n"
        "       qdt2_2mEz[id]=co[1]*qdt2_2mEz[id]; \n"
        "       px[id]=co[0]*px[id]  ;                        \n"
        "       py[id]=co[0]*py[id]  ;                        \n"
        "       pz[id]=co[0]*pz[id]  ;                        \n"
        "       pxx[id]=px[id]*px[id];      \n"
        "       pxy[id]=px[id]*py[id];      \n"
        "       pxz[id]=px[id]*pz[id];      \n"
        "       pyy[id]=py[id]*py[id];      \n"
        "       pyz[id]=py[id]*pz[id];      \n"
        "       pzz[id]=pz[id]*pz[id];      \n"
        "       r_determinant[id]=1.0/(1.0+ pxx[id]+pyy[id]+pzz[id]); \n"//r_determinant[n] = 1.0/(1.0+pxx[n]+pyy[n]+pzz[n]);
        "       dx[id]=2.0*pos1x[id]-pos0x[id]-pz[id]*pos0y[id]+py[id]*pos0z[id]; \n"// dx[n]=2*pos1x[p][n]-pos0x[p][n]-pz[n]*pos0y[p][n]+py[n]*pos0z[p][n];
        "       dy[id]=2.0*pos1y[id]-pos0y[id]-px[id]*pos0z[id]+pz[id]*pos0x[id]; \n"// dy[n]=2*pos1y[p][n]-pos0y[p][n]-px[n]*pos0z[p][n]+pz[n]*pos0x[p][n];
        "       dz[id]=2.0*pos1z[id]-pos0z[id]-py[id]*pos0x[id]+px[id]*pos0y[id]; \n"// dz[n]=2*pos1z[p][n]-pos0z[p][n]-py[n]*pos0x[p][n]+px[n]*pos0y[p][n];
        "       M1[id]=(1.0    +pxx[id])*dx[id]; "//M1[n]=(1.0    +pxx[n])*dx[n];
        "       M2[id]=(pz[id] +pxy[id])*dy[id]; "//M2[n]=(pz[n]  +pxy[n])*dy[n];
        "       M3[id]=(pxz[id]-py[id] )*dz[id]; " //M3[n]=(pxz[n] -py[n] )*dz[n];
        "       pos0x[id]=r_determinant[id]*(M1[id]+M2[id]+M3[id])+qdt2_2mEx[id]; "//pos1x[p][n]=r_determinant[n]*(M1[n]+M2[n]+M3[n]) +qdt2_2mEx[n];
        "       M1[id]=(pxy[id]-pz[id] )*dx[id]; "//M1[n]=(pxy[n] -pz[n] )*dx[n];
        "       M2[id]=(1.0    +pyy[id])*dy[id]; "//M2[n]=(1.0    +pyy[n])*dy[n];
        "       M3[id]=(pyz[id]+px[id] )*dz[id]; "//M3[n]=(pyz[n] +px[n] )*dz[n];
        "       pos0y[id]=r_determinant[id]*(M1[id]+M2[id]+M3[id])+qdt2_2mEy[id]; " //pos1y[p][n]=r_determinant[n]*(M1[n]+M2[n]+M3[n]) +qdt2_2mEy[n];
        "       M1[id]=(pxz[id]+py[id] )*dx[id]; "//M1[n]=(pxz[n] +py[n] )*dx[n];
        "       M3[id]=(pyz[id]-px[id] )*dy[id]; "//M2[n]=(pyz[n] -px[n] )*dy[n];
        "       M2[id]=(1.0    +pzz[id])*dz[id]; "//M3[n]=(1.0    +pzz[n])*dz[n];
        "       pos0z[id]=r_determinant[id]*(M1[id]+M2[id]+M3[id])+qdt2_2mEz[id]; "//pos1z[p][n]=r_determinant[n]*(M1[n]+M2[n]+M3[n]) +qdt2_2mEz[n];
        //  "       if (pos0x[id]>co[2]) {pos0x[id]=co[2]; pos1x=co[2];}"
        "   }";
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS)
    {
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    context1=context;
    default_device1=default_device;
    program1=program;
}

void calc_pnn(float *px,float *py,float *pz,
              float *qdt2_2mEx,float *qdt2_2mEy,float *qdt2_2mEz,
              float *pos0x,float *pos0y,float *pos0z,
              float *pos1x,float *pos1y, float *pos1z,
              float *qdt_m,
              int n, cl::Context &context,cl::Device &default_device,cl::Program &program)
{
    // create buffers on the device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_D(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_E(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_F(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_G(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_H(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_I(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_J(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_K(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_L(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_M(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_N(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_O(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_P(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_Q(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_R(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_S(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_T(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_U(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_V(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_W(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_X(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_Y(context,CL_MEM_READ_WRITE,sizeof(float)*n);
    cl::Buffer buffer_Z(context,CL_MEM_READ_WRITE,sizeof(float)*2);

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    //write input arrays to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(float)*n,px);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(float)*n,py);
    queue.enqueueWriteBuffer(buffer_C,CL_TRUE,0,sizeof(float)*n,pz);
    queue.enqueueWriteBuffer(buffer_D,CL_TRUE,0,sizeof(float)*n,qdt2_2mEx);
    queue.enqueueWriteBuffer(buffer_E,CL_TRUE,0,sizeof(float)*n,qdt2_2mEy);
    queue.enqueueWriteBuffer(buffer_F,CL_TRUE,0,sizeof(float)*n,qdt2_2mEz);
    queue.enqueueWriteBuffer(buffer_G,CL_TRUE,0,sizeof(float)*n,pos0x);
    queue.enqueueWriteBuffer(buffer_H,CL_TRUE,0,sizeof(float)*n,pos0y);
    queue.enqueueWriteBuffer(buffer_I,CL_TRUE,0,sizeof(float)*n,pos0z);
    queue.enqueueWriteBuffer(buffer_J,CL_TRUE,0,sizeof(float)*n,pos1x);
    queue.enqueueWriteBuffer(buffer_K,CL_TRUE,0,sizeof(float)*n,pos1y);
    queue.enqueueWriteBuffer(buffer_L,CL_TRUE,0,sizeof(float)*n,pos1z);
    queue.enqueueWriteBuffer(buffer_Z,CL_TRUE,0,sizeof(float)*2,qdt_m);

    //run the kernel
    cl::Kernel kernel_add=cl::Kernel(program,"calc_pnn_k");//select the kernel program to run
    kernel_add.setArg(0,buffer_A); //the 1st argument to the kernel program
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    kernel_add.setArg(3,buffer_D);
    kernel_add.setArg(4,buffer_E);
    kernel_add.setArg(5,buffer_F);
    kernel_add.setArg(6,buffer_G);
    kernel_add.setArg(7,buffer_H);
    kernel_add.setArg(8,buffer_I);
    kernel_add.setArg(9,buffer_J);
    kernel_add.setArg(10,buffer_K);
    kernel_add.setArg(11,buffer_L);
    kernel_add.setArg(12,buffer_M);
    kernel_add.setArg(13,buffer_N);
    kernel_add.setArg(14,buffer_O);
    kernel_add.setArg(15,buffer_P);
    kernel_add.setArg(16,buffer_Q);
    kernel_add.setArg(17,buffer_R);
    kernel_add.setArg(18,buffer_S);
    kernel_add.setArg(19,buffer_T);
    kernel_add.setArg(20,buffer_U);
    kernel_add.setArg(21,buffer_V);
    kernel_add.setArg(22,buffer_W);
    kernel_add.setArg(23,buffer_X);
    kernel_add.setArg(24,buffer_Y);
    kernel_add.setArg(25,buffer_Z);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(n),cl::NullRange);
    queue.finish(); //wait for the end of the kernel program

    //read result arrays from the device to main memory
    queue.enqueueReadBuffer(buffer_G,CL_TRUE,0,sizeof(float)*n,pos1x);
    queue.enqueueReadBuffer(buffer_H,CL_TRUE,0,sizeof(float)*n,pos1y);
    queue.enqueueReadBuffer(buffer_I,CL_TRUE,0,sizeof(float)*n,pos1z);
    queue.enqueueReadBuffer(buffer_J,CL_TRUE,0,sizeof(float)*n,pos0x);
    queue.enqueueReadBuffer(buffer_K,CL_TRUE,0,sizeof(float)*n,pos0y);
    queue.enqueueReadBuffer(buffer_L,CL_TRUE,0,sizeof(float)*n,pos0z);
}

