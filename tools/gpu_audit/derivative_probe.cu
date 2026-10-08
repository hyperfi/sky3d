// Standalone FP64 spectral-derivative probe, not a GPU TDHF implementation.
// CPU plans/call granularity follow Code/fourier.f90 and Code/levels.f90.
#include <cuda_runtime.h>
#include <cufft.h>
#include <fftw3.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1); }} while(0)
#define FFT(call) do { auto e=(call); if(e!=CUFFT_SUCCESS) { \
  std::fprintf(stderr,"cuFFT %s:%d: %d\n",__FILE__,__LINE__,int(e)); std::exit(1); }} while(0)
using C = cufftDoubleComplex;
using Clock = std::chrono::steady_clock;

__device__ size_t physical(size_t t, int n, int axis) {
  size_t k=t%n, line=t/n, outer=line/(n*n);
  size_t x=axis==0?k:line%n;
  size_t y=axis==1?k:(axis==0?line%n:(line/n)%n);
  size_t z=axis==2?k:(line/n)%n;
  return x+n*(y+n*(z+n*outer));
}
__global__ void pack(const C* input,C* packed,size_t count,int n,int axis) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<count) packed[t]=input[physical(t,n,axis)];
}
__global__ void multiply(C* p,size_t count,int n) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t>=count) return;
  int k=t%n;
  double factor=(k==n/2?0.0:(k<n/2?k:k-n))*(2.0*3.14159265358979)/(n*n);
  C v=p[t]; p[t]={-factor*v.y,factor*v.x};
}
__global__ void unpack(const C* packed,C* output,size_t count,int n,int axis) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<count) output[physical(t,n,axis)]=packed[t];
}

struct Probe {
  int n,states;
  size_t spinor,count,bytes;
  C *input,*cpu,*hostgpu,*deviceinput,*deviceout,*scratch;
  fftw_plan forward[3], backward[3];
  cufftHandle gpuplan;
  Probe(int points,int orbitals):n(points),states(orbitals) {
    spinor=size_t(n)*n*n*2; count=spinor*states; bytes=count*sizeof(C);
    input=(C*)fftw_malloc(bytes); cpu=(C*)fftw_malloc(bytes*3); hostgpu=(C*)fftw_malloc(bytes*3);
    if(!input||!cpu||!hostgpu) std::exit(1);
    for(int axis=0;axis<3;axis++) {
      int stride=axis==0?1:(axis==1?n:n*n);
      int distance=axis==0?n:1;
      int batch=axis==0?2*n*n:(axis==1?n:n*n);
      forward[axis]=fftw_plan_many_dft(1,&n,batch,(fftw_complex*)input,nullptr,stride,distance,
          (fftw_complex*)(cpu+axis*count),nullptr,stride,distance,FFTW_FORWARD,FFTW_PATIENT);
      backward[axis]=fftw_plan_many_dft(1,&n,batch,(fftw_complex*)(cpu+axis*count),nullptr,stride,distance,
          (fftw_complex*)(cpu+axis*count),nullptr,stride,distance,FFTW_BACKWARD,FFTW_PATIENT);
      if(!forward[axis]||!backward[axis]) std::exit(1);
    }
    // Deterministic nonsymmetric data exercise both complex parts and all modes.
    for(size_t i=0;i<count;i++) input[i]={std::sin(i*0.017)+0.3*std::cos(i*0.13),std::cos(i*0.021)};
    CUDA(cudaMalloc(&deviceinput,bytes)); CUDA(cudaMalloc(&deviceout,bytes*3)); CUDA(cudaMalloc(&scratch,bytes));
    FFT(cufftPlan1d(&gpuplan,n,CUFFT_Z2Z,2*n*n*states));
    CUDA(cudaMemcpy(deviceinput,input,bytes,cudaMemcpyHostToDevice));
  }
  ~Probe() {
    for(int a=0;a<3;a++) {fftw_destroy_plan(forward[a]);fftw_destroy_plan(backward[a]);}
    FFT(cufftDestroy(gpuplan)); CUDA(cudaFree(deviceinput)); CUDA(cudaFree(deviceout)); CUDA(cudaFree(scratch));
    fftw_free(input);fftw_free(cpu);fftw_free(hostgpu);
  }
  void cpu_derivatives(int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int state=0;state<states;state++) {
      for(int a=0;a<3;a++) {
        C* src=input+state*spinor; C* out=cpu+a*count+state*spinor;
        int chunk=a==0?int(spinor):(a==1?n*n:n*n*n);
        for(size_t offset=0;offset<spinor;offset+=chunk)
          fftw_execute_dft(forward[a],(fftw_complex*)(src+offset),(fftw_complex*)(out+offset));
        for(size_t i=0;i<spinor;i++) {
          int k=(i/(a==0?1:(a==1?n:n*n)))%n;
          double factor=(k==n/2?0.0:(k<n/2?k:k-n))*(2.0*3.14159265358979)/(n*n);
          C v=out[i]; out[i]={-factor*v.y,factor*v.x};
        }
        for(size_t offset=0;offset<spinor;offset+=chunk)
          fftw_execute_dft(backward[a],(fftw_complex*)(out+offset),(fftw_complex*)(out+offset));
      }
    }
  }
  void gpu_derivatives() {
    int blocks=(count+255)/256;
    for(int a=0;a<3;a++) {
      pack<<<blocks,256>>>(deviceinput,scratch,count,n,a);
      FFT(cufftExecZ2Z(gpuplan,scratch,scratch,CUFFT_FORWARD));
      multiply<<<blocks,256>>>(scratch,count,n);
      FFT(cufftExecZ2Z(gpuplan,scratch,scratch,CUFFT_INVERSE));
      unpack<<<blocks,256>>>(scratch,deviceout+a*count,count,n,a);
    }
    CUDA(cudaGetLastError());
  }
  void copy_output() {CUDA(cudaMemcpy(hostgpu,deviceout,bytes*3,cudaMemcpyDeviceToHost));}
};

template<class F> double median_ms(F fn,int reps) {
  double times[3];
  for(int trial=0;trial<3;trial++) {
    auto start=Clock::now();
    for(int r=0;r<reps;r++) fn();
    times[trial]=std::chrono::duration<double,std::milli>(Clock::now()-start).count()/reps;
  }
  std::sort(times,times+3);return times[1];
}
int main(int argc,char** argv) {
  int n=argc>1?std::atoi(argv[1]):24,states=argc>2?std::atoi(argv[2]):20,reps=argc>3?std::atoi(argv[3]):10;
  if(n<2||n%2||states<1||reps<1) return 2;
  CUDA(cudaSetDevice(0));
  Probe p(n,states);
  p.cpu_derivatives(1);p.gpu_derivatives();CUDA(cudaDeviceSynchronize());p.copy_output();
  long double difference=0,norm=0; double max_error=0;
  for(size_t i=0;i<3*p.count;i++) {
    double dx=p.cpu[i].x-p.hostgpu[i].x,dy=p.cpu[i].y-p.hostgpu[i].y;
    difference+=(long double)dx*dx+(long double)dy*dy;
    norm+=(long double)p.cpu[i].x*p.cpu[i].x+(long double)p.cpu[i].y*p.cpu[i].y;
    max_error=std::max(max_error,std::hypot(dx,dy));
  }
  double error=std::sqrt(difference/norm);
  if(!std::isfinite(error)||error>1e-11) {std::fprintf(stderr,"Derivative validation failed: %.4e\n",error);return 1;}
  double resident=median_ms([&]{p.gpu_derivatives();CUDA(cudaDeviceSynchronize());},reps);
  double transfer=median_ms([&]{CUDA(cudaMemcpy(p.deviceinput,p.input,p.bytes,cudaMemcpyHostToDevice));
                              p.gpu_derivatives();p.copy_output();},reps);
  for(int threads: {1,8}) {
    p.cpu_derivatives(threads);
    double cpu=median_ms([&]{p.cpu_derivatives(threads);},reps);
    std::printf("%d,%d,%d,%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.5e,%.5e\n",
      n,states,reps,threads,cpu,resident,transfer,cpu/resident,cpu/transfer,error,max_error);
  }
}
