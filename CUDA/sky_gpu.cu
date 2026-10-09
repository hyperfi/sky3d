// Single-process FP64 TDHF propagation/density backend. No managed-memory spill.
#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Z=cufftDoubleComplex;
static constexpr double PI=3.14159265358979;
#define CUDA(x) do {auto e=(x);if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));} while(0)
#define FFT(x) do {auto e=(x);if(e!=CUFFT_SUCCESS) throw std::runtime_error(std::string(#x)+": cuFFT error "+std::to_string(e));} while(0)
__host__ __device__ inline Z operator+(Z a,Z b){return {a.x+b.x,a.y+b.y};}
__host__ __device__ inline Z operator-(Z a,Z b){return {a.x-b.x,a.y-b.y};}
__host__ __device__ inline Z operator*(Z a,Z b){return {a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};}
__host__ __device__ inline Z operator*(double a,Z b){return {a*b.x,a*b.y};}
__device__ inline Z conjugate(Z a){return {a.x,-a.y};}
__device__ inline double norm2(Z a){return a.x*a.x+a.y*a.y;}
__device__ inline int field_index(int cell,int axis,int iq,int g){return cell+g*(axis+3*iq);}

__device__ int physical_index(int t,int nx,int ny,int nz,int axis){
  int n=axis==0?nx:(axis==1?ny:nz),k=t%n,line=t/n;
  int x=axis==0?k:line%nx;
  int y=axis==1?k:(axis==0?line%ny:(line/nx)%ny);
  int z=axis==2?k:(axis==0?(line/ny)%nz:(line/nx)%nz);
  int outer=line/(axis==0?ny*nz:(axis==1?nx*nz:nx*ny));
  return x+nx*(y+ny*(z+nz*outer));
}
__global__ void pack_axis(const Z* input,Z* packed,int count,int nx,int ny,int nz,int axis){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<count) packed[t]=input[physical_index(t,nx,ny,nz,axis)];
}
__global__ void unpack_axis(const Z* packed,Z* output,int count,int nx,int ny,int nz,int axis){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<count) output[physical_index(t,nx,ny,nz,axis)]=packed[t];
}
__global__ void spectral_scale(Z* first,Z* second,int count,int n,double spacing,int order){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=count) return;
  int k=t%n,mode=k<n/2?k:k-n;
  double wave=mode*(2.0*PI)/(n*spacing);
  if(order==2) second[t]=(-wave*wave/n)*first[t];
  else first[t]=(k==n/2?0.0:wave/n)*Z{-first[t].y,first[t].x};
}
__global__ void h_local(const Z* p,Z* h,int count,int g,const int* iq,
                        const double* u,const double* s){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=count)return;
  int cell=t%g,spin=(t/g)%2,q=iq[t/(2*g)]-1;
  int other=t+(spin==0?g:-g),base=cell+g*q;
  double sx=s[field_index(cell,0,q,g)],sy=s[field_index(cell,1,q,g)],sz=s[field_index(cell,2,q,g)];
  h[t]=u[base]*p[t]+Z{sx,spin==0?-sy:sy}*p[other]+(spin==0?sz:-sz)*p[t];
}
__global__ void h_axis(const Z* p,const Z* d1,const Z* d2,Z* h,Z* coupled,
                       int count,int g,const int* iq,int axis,const double* b,
                       const double* dbm,const double* a,const double* w){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=count)return;
  int cell=t%g,spin=(t/g)%2,q=iq[t/(2*g)]-1,other=t+(spin==0?g:-g);
  double sign=spin==0?0.5:-0.5,B=b[cell+g*q],D=dbm[field_index(cell,axis,q,g)],A=a[field_index(cell,axis,q,g)];
  double wx=w[field_index(cell,0,q,g)],wy=w[field_index(cell,1,q,g)],wz=w[field_index(cell,2,q,g)];
  Z value=h[t]-B*d2[t];
  if(axis==0){
    value=value-Z{D,0.5*A-sign*wy}*d1[t]-(sign*wz)*d1[other];
    coupled[t]=Z{0,-0.5*(A-2*sign*wy)}*p[t]-(sign*wz)*p[other];
  }else if(axis==1){
    value=value-Z{D,0.5*A+sign*wx}*d1[t]+Z{0,0.5*wz}*d1[other];
    coupled[t]=Z{0,-0.5*(A+2*sign*wx)}*p[t]+Z{0,0.5*wz}*p[other];
  }else{
    value=value-Z{D,0.5*A}*d1[t]+Z{sign*wx,-0.5*wy}*d1[other];
    coupled[t]=Z{0,-0.5*A}*p[t]+Z{sign*wx,-0.5*wy}*p[other];
  }
  h[t]=value;
}
__global__ void add_h(Z* h,const Z* d,int count){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<count)h[t]=h[t]+d[t];
}
__global__ void recurrence(Z* term,Z* result,const Z* h,int count,double coefficient){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<count){Z v={-coefficient*h[t].y,coefficient*h[t].x};term[t]=v;result[t]=result[t]+v;}
}
__global__ void density_kernel(const Z* p,const Z* gx,const Z* gy,const Z* gz,
                               double* den,int g,int states,const int* iq,const double* weights){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=2*g)return;
  int cell=t%g,q=t/g;double v[11]={0};
  for(int state=0;state<states;state++){
    double weight=weights[state];if(iq[state]-1!=q||weight<=0)continue;
    int index=cell+state*2*g;Z p0=p[index],p1=p[index+g];
    Z cross=conjugate(p0)*p1;
    v[0]+=weight*(norm2(p0)+norm2(p1));
    v[5]+=2*weight*cross.x;v[6]+=2*weight*cross.y;v[7]+=weight*(norm2(p0)-norm2(p1));
    const Z* grads[3]={gx,gy,gz};
    for(int axis=0;axis<3;axis++){
      Z d0=grads[axis][index],d1=grads[axis][index+g];
      Z a=d0*conjugate(p0),b=d1*conjugate(p1),c=d0*conjugate(p1),d=d1*conjugate(p0);
      v[1]+=weight*(norm2(d0)+norm2(d1));v[2+axis]+=weight*(a.y+b.y);
      if(axis==0){v[9]-=weight*(a.y-b.y);v[10]+=weight*(c.x-d.x);}
      else if(axis==1){v[8]+=weight*(a.y-b.y);v[10]-=weight*(c.y+d.y);}
      else{v[8]+=weight*(d.x-c.x);v[9]+=weight*(c.y+d.y);}
    }
  }
  for(int component=0;component<11;component++)den[cell+g*(component+11*q)]=v[component];
}

// Real density derivatives use the same Fourier collocation convention as
// Grids.sder/sder2: first-derivative Nyquist zero, second derivative retained.
__global__ void pack_real(const double* src,Z* dst,int count,int nx,int ny,int nz,
                          int axis,int channels,int component){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;
  int g=nx*ny*nz,index=physical_index(t,nx,ny,nz,axis);
  dst[t]={src[index%g+g*(component+channels*(index/g))],0};
}
__global__ void unpack_real(const Z* src,double* dst,int count,int nx,int ny,int nz,
                            int axis,int channels,int component,double factor,bool add){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;
  int g=nx*ny*nz,index=physical_index(t,nx,ny,nz,axis);
  int out=index%g+g*(component+channels*(index/g));
  double value=factor*src[t].x;dst[out]=add?dst[out]+value:value;
}
__global__ void real_scale(Z* v,int count,int n,double spacing,int order){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;
  int k=t%n,mode=k<n/2?k:k-n;double wave=mode*(2*PI)/(n*spacing);
  v[t]=order==2?(-wave*wave/n)*v[t]:(k==n/2?0.0:wave/n)*Z{-v[t].y,v[t].x};
}
struct Couplings{double c[15];};
__global__ void skyrme_local(const double* den,const double* divj,const double* lap,
  const double* grad,const double* curls,const double* curlc,const double* wc,
  double* u,double* b,double* s,double* a,double* w,int g,Couplings f,bool coul){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=2*g)return;
  int cell=t%g,q=t/g,other=1-q;const double* c=f.c;
  double rq=den[cell+g*(11*q)],rc=den[cell+g*(11*other)],total=rq+rc;
  double nonlinear=pow(total,c[10])*((c[6]*(c[10]+2)/3-2*c[7]/3)*rq
    +c[6]*(c[10]+2)/3*rc-(c[7]*c[10]/3)*(rq*rq+rc*rc)/(total+1e-25));
  u[t]=nonlinear-(c[8]+c[9])*divj[t]-c[8]*divj[cell+g*other]
    +(c[0]-c[1])*rq+c[0]*rc+(c[2]-c[3])*den[cell+g*(1+11*q)]
    +c[2]*den[cell+g*(1+11*other)]-(c[4]-c[5])*lap[t]-c[4]*lap[cell+g*other];
  if(coul&&q==1){u[t]+=wc[cell];if(c[14]!=0)u[t]-=c[11]*pow(rq,1.0/3.0);}
  b[t]=c[12+q]+(c[2]-c[3])*rq+c[2]*rc;
  for(int k=0;k<3;k++){
    int i=cell+g*(k+3*q),j=cell+g*(k+3*other);
    w[i]=(c[8]+c[9])*grad[i]+c[8]*grad[j];
    a[i]=-2*(c[2]-c[3])*den[cell+g*(2+k+11*q)]-2*c[2]*den[cell+g*(2+k+11*other)]
      -(c[8]+c[9])*curls[i]-c[8]*curls[j];
    s[i]=-(c[8]+c[9])*curlc[i]-c[8]*curlc[j];
  }
}
__global__ void coulomb_pad(const double* den,Z* out,int nx,int ny,int nz,int factor){
  int t=blockIdx.x*blockDim.x+threadIdx.x,g=nx*ny*nz;
  int xdim=nx*factor,ydim=ny*factor,zdim=nz*factor;
  if(t>=xdim*ydim*zdim)return;
  int x=t%xdim,y=(t/xdim)%ydim,z=t/(xdim*ydim);
  out[t]=(x<nx&&y<ny&&z<nz)?Z{den[x+nx*(y+ny*z)+11*g],0}:Z{0,0};
}
__global__ void coulomb_product(Z* data,const Z* kernel,int count,double scale){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<count)data[t]=scale*(kernel[t]*data[t]);
}
__global__ void coulomb_extract(const Z* data,double* wc,int nx,int ny,int nz,int factor){
  int t=blockIdx.x*blockDim.x+threadIdx.x,g=nx*ny*nz;if(t>=g)return;
  int x=t%nx,y=(t/nx)%ny,z=t/(nx*ny);
  wc[t]=data[x+nx*factor*(y+ny*factor*z)].x/(g*factor*factor*factor);
}

struct PlanSet{
  cufftHandle plans[3]={0,0,0};
  cufftHandle realplans[3]={0,0,0},coulplans[2]={0,0};
  size_t workspace=0;int nx,ny,nz,states,g,count;
  PlanSet(int x,int y,int z,int ns):nx(x),ny(y),nz(z),states(ns){
    if(x<2||y<2||z<2||x%2||y%2||z%2||ns<1)throw std::runtime_error("Grid dimensions must be positive even integers and states positive");
    uint64_t cells=uint64_t(x)*uint64_t(y)*uint64_t(z);
    if(cells>INT_MAX/22||uint64_t(ns)>uint64_t(INT_MAX)/(2*cells))throw std::runtime_error("Case exceeds current single-batch indexing limits");
    g=int(cells);count=int(2*cells*ns);
    try{for(int axis=0;axis<3;axis++){
      int n=axis==0?x:(axis==1?y:z);size_t bytes=0;
      FFT(cufftCreate(&plans[axis]));FFT(cufftSetAutoAllocation(plans[axis],0));
      FFT(cufftMakePlan1d(plans[axis],n,CUFFT_Z2Z,count/n,&bytes));workspace=std::max(workspace,bytes);
    }
    for(int axis=0;axis<3;axis++){
      int n=axis==0?x:(axis==1?y:z);size_t bytes=0;
      FFT(cufftCreate(&realplans[axis]));FFT(cufftSetAutoAllocation(realplans[axis],0));
      FFT(cufftMakePlan1d(realplans[axis],n,CUFFT_Z2Z,2*g/n,&bytes));workspace=std::max(workspace,bytes);
    }
    for(int mode=0;mode<2;mode++){
      int factor=mode+1;size_t bytes=0;
      FFT(cufftCreate(&coulplans[mode]));FFT(cufftSetAutoAllocation(coulplans[mode],0));
      FFT(cufftMakePlan3d(coulplans[mode],factor*z,factor*y,factor*x,CUFFT_Z2Z,&bytes));
      workspace=std::max(workspace,bytes);
    }
    }catch(...){for(auto p:plans)if(p)cufftDestroy(p);for(auto p:realplans)if(p)cufftDestroy(p);
      for(auto p:coulplans)if(p)cufftDestroy(p);throw;}
  }
  ~PlanSet(){for(auto p:plans)if(p)cufftDestroy(p);for(auto p:realplans)if(p)cufftDestroy(p);
    for(auto p:coulplans)if(p)cufftDestroy(p);}
  uint64_t required()const{return uint64_t(count)*sizeof(Z)*9+uint64_t(g)*840+uint64_t(states)*12+workspace;}
};
struct DeviceMemory{
  std::vector<void*> pointers;
  ~DeviceMemory(){for(void* p:pointers)cudaFree(p);}
};
struct StreamOwner{
  cudaStream_t value;
  StreamOwner(){CUDA(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
  ~StreamOwner(){cudaStreamDestroy(value);}
};
using GraphKey=std::tuple<int,double,uintptr_t>;
struct GraphCache{
  std::map<GraphKey,cudaGraphExec_t> entries;
  ~GraphCache(){for(auto& pair:entries)cudaGraphExecDestroy(pair.second);}
};
struct State:PlanSet{
  DeviceMemory allocations;
  StreamOwner execution;
  cudaStream_t stream=execution.value;
  GraphCache graphs;
  Z *psi,*result,*term,*h,*d1,*d2,*coupled,*frequency,*second;
  double *u,*b,*s,*dbm,*a,*w,*den,*weights,spacing[3];int* iq;
  double *divj,*laprho,*gradrho,*curls,*curlc,*wc;
  Z *coulwork,*coulkernel;
  bool coul_enabled=false,periodic=false,device_fields=false;
  double coul_scale=0;
  void* work;
  template<class T>void alloc(T*& p,size_t bytes){CUDA(cudaMalloc(&p,bytes));
    try{allocations.pointers.push_back(p);}catch(...){cudaFree(p);throw;}}
  State(int x,int y,int z,int ns,const double* steps,const Z* p,const double* occ,const int* isospin):PlanSet(x,y,z,ns){
    size_t free=0,total=0;CUDA(cudaMemGetInfo(&free,&total));
    double fraction=0.80;const char* setting=std::getenv("SKY3D_GPU_MEMORY_FRACTION");
    if(setting){char* end=nullptr;fraction=std::strtod(setting,&end);
      if(!end||*end||!std::isfinite(fraction)||fraction<0.05||fraction>0.95)throw std::runtime_error("SKY3D_GPU_MEMORY_FRACTION must be between 0.05 and 0.95");}
    std::fprintf(stdout,"GPU memory: required %.1f MiB incl cuFFT workspace; free %.1f MiB, total %.1f MiB\n",required()/1048576.,free/1048576.,total/1048576.);
    if(required()>free*fraction)throw std::runtime_error("Insufficient free VRAM with safety headroom. Use SKY3D_BACKEND=cpu. Oversubscription/extra transfers can be slower than parallel CPU; this backend does not page GPU arrays to system RAM.");
    if(required()>free*0.60)std::fprintf(stderr,"WARNING: limited VRAM headroom; other GPU use may cause allocation failure. Compare a parallel CPU run before assuming a speedup.\n");
    for(int k=0;k<3;k++){if(!std::isfinite(steps[k])||steps[k]<=0)throw std::runtime_error("Grid spacing must be positive");spacing[k]=steps[k];}
    for(int k=0;k<states;k++)if(isospin[k]<1||isospin[k]>2||!std::isfinite(occ[k]))throw std::runtime_error("Invalid occupation or isospin");
    size_t bytes=size_t(count)*sizeof(Z);
    for(Z** p:{&psi,&result,&term,&h,&d1,&d2,&coupled,&frequency,&second})alloc(*p,bytes);
    alloc(u,size_t(g)*2*8);alloc(b,size_t(g)*2*8);
    for(double** p:{&s,&dbm,&a,&w})alloc(*p,size_t(g)*6*8);
    alloc(den,size_t(g)*22*8);alloc(weights,size_t(states)*8);alloc(iq,size_t(states)*4);
    alloc(divj,size_t(g)*2*8);alloc(laprho,size_t(g)*2*8);
    for(double** p:{&gradrho,&curls,&curlc})alloc(*p,size_t(g)*6*8);
    alloc(wc,size_t(g)*8);alloc(coulwork,size_t(g)*8*sizeof(Z));alloc(coulkernel,size_t(g)*8*sizeof(Z));
    work=nullptr;if(workspace)alloc(work,workspace);
    for(auto plan:plans){FFT(cufftSetWorkArea(plan,work));FFT(cufftSetStream(plan,stream));}
    for(auto plan:realplans){FFT(cufftSetWorkArea(plan,work));FFT(cufftSetStream(plan,stream));}
    for(auto plan:coulplans){FFT(cufftSetWorkArea(plan,work));FFT(cufftSetStream(plan,stream));}
    CUDA(cudaMemcpy(psi,p,bytes,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(weights,occ,size_t(states)*8,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(iq,isospin,size_t(states)*4,cudaMemcpyHostToDevice));
    std::fflush(stdout);
  }
  int blocks()const{return (count-1)/256+1;}
  void real_derivative(const double* src,int channels,int component,int axis,int order,
                       double* dst,int outchannels,int outcomponent,double factor=1,bool add=false){
    int n=axis==0?nx:(axis==1?ny:nz),blocks=(2*g-1)/256+1;
    pack_real<<<blocks,256,0,stream>>>(src,frequency,2*g,nx,ny,nz,axis,channels,component);
    FFT(cufftExecZ2Z(realplans[axis],frequency,frequency,CUFFT_FORWARD));
    real_scale<<<blocks,256,0,stream>>>(frequency,2*g,n,spacing[axis],order);
    FFT(cufftExecZ2Z(realplans[axis],frequency,frequency,CUFFT_INVERSE));
    unpack_real<<<blocks,256,0,stream>>>(frequency,dst,2*g,nx,ny,nz,axis,outchannels,outcomponent,factor,add);
  }
  void curl(int component,double* output){
    for(int axis=0;axis<3;axis++){
      int j=(axis+1)%3,k=(axis+2)%3;
      real_derivative(den,11,component+k,j,1,output,3,axis);
      real_derivative(den,11,component+j,k,1,output,3,axis,-1,true);
    }
  }
  void skyrme(Couplings f){
    for(int axis=0;axis<3;axis++){
      real_derivative(den,11,8+axis,axis,1,divj,1,0,1,axis!=0);
      real_derivative(den,11,0,axis,2,laprho,1,0,1,axis!=0);
      real_derivative(den,11,0,axis,1,gradrho,3,axis);
    }
    curl(5,curls);curl(2,curlc);
    if(coul_enabled){
      int factor=periodic?1:2,n=g*factor*factor*factor;
      coulomb_pad<<<(n-1)/256+1,256,0,stream>>>(den,coulwork,nx,ny,nz,factor);
      FFT(cufftExecZ2Z(coulplans[factor-1],coulwork,coulwork,CUFFT_FORWARD));
      coulomb_product<<<(n-1)/256+1,256,0,stream>>>(coulwork,coulkernel,n,coul_scale);
      FFT(cufftExecZ2Z(coulplans[factor-1],coulwork,coulwork,CUFFT_INVERSE));
      coulomb_extract<<<(g-1)/256+1,256,0,stream>>>(coulwork,wc,nx,ny,nz,factor);
    }
    skyrme_local<<<(2*g-1)/256+1,256,0,stream>>>(den,divj,laprho,gradrho,curls,curlc,wc,u,b,s,a,w,g,f,coul_enabled);
    for(int axis=0;axis<3;axis++)real_derivative(b,1,0,axis,1,dbm,3,axis);
    CUDA(cudaGetLastError());
  }
  void derivative(const Z* input,int axis,Z* first,Z* second_output=nullptr){
    int n=axis==0?nx:(axis==1?ny:nz);
    pack_axis<<<blocks(),256,0,stream>>>(input,frequency,count,nx,ny,nz,axis);
    FFT(cufftExecZ2Z(plans[axis],frequency,frequency,CUFFT_FORWARD));
    if(second_output){
      spectral_scale<<<blocks(),256,0,stream>>>(frequency,second,count,n,spacing[axis],2);
      FFT(cufftExecZ2Z(plans[axis],second,second,CUFFT_INVERSE));
      unpack_axis<<<blocks(),256,0,stream>>>(second,second_output,count,nx,ny,nz,axis);
    }
    spectral_scale<<<blocks(),256,0,stream>>>(frequency,nullptr,count,n,spacing[axis],1);
    FFT(cufftExecZ2Z(plans[axis],frequency,frequency,CUFFT_INVERSE));
    unpack_axis<<<blocks(),256,0,stream>>>(frequency,first,count,nx,ny,nz,axis);
  }
  void hpsi(const Z* input){
    h_local<<<blocks(),256,0,stream>>>(input,h,count,g,iq,u,s);
    for(int axis=0;axis<3;axis++){
      derivative(input,axis,d1,d2);
      h_axis<<<blocks(),256,0,stream>>>(input,d1,d2,h,coupled,count,g,iq,axis,b,dbm,a,w);
      derivative(coupled,axis,d1);
      add_h<<<blocks(),256,0,stream>>>(h,d1,count);
    }
  }
  void densities(const Z* input){
    derivative(input,0,d1);derivative(input,1,d2);derivative(input,2,coupled);
    density_kernel<<<(2*g-1)/256+1,256,0,stream>>>(input,d1,d2,coupled,den,g,states,iq,weights);
  }
  void enqueue_stage(int order,double dt_hbc){
    size_t bytes=size_t(count)*sizeof(Z);
    CUDA(cudaMemcpyAsync(result,psi,bytes,cudaMemcpyDeviceToDevice,stream));
    CUDA(cudaMemcpyAsync(term,psi,bytes,cudaMemcpyDeviceToDevice,stream));
    for(int m=1;m<=order;m++){
      hpsi(term);recurrence<<<blocks(),256,0,stream>>>(term,result,h,count,-dt_hbc/m);
    }
    densities(result);CUDA(cudaGetLastError());
  }
  void stage(int order,double dt_hbc){
    const char* setting=std::getenv("SKY3D_GPU_GRAPHS");
    if(setting&&std::string(setting)=="0"){enqueue_stage(order,dt_hbc);return;}
    if(setting&&std::string(setting)!="1")throw std::runtime_error("SKY3D_GPU_GRAPHS must be 0 or 1");
    // Each alternating wavefunction buffer and Taylor order has its own graph.
    // Field values change in place; timestep changes select a different key.
    GraphKey key{order,dt_hbc,reinterpret_cast<uintptr_t>(psi)};
    auto found=graphs.entries.find(key);
    if(found==graphs.entries.end()){
      cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
      CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      try{enqueue_stage(order,dt_hbc);}catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
      CUDA(cudaStreamEndCapture(stream,&graph));
      try{CUDA(cudaGraphInstantiate(&executable,graph,0));}catch(...){cudaGraphDestroy(graph);throw;}
      cudaGraphDestroy(graph);
      try{found=graphs.entries.emplace(key,executable).first;}catch(...){cudaGraphExecDestroy(executable);throw;}
    }
    CUDA(cudaGraphLaunch(found->second,stream));
  }
};
static void failure(const std::exception& e){std::fprintf(stderr,"Sky3D GPU: %s\n",e.what());std::exit(3);}

// The query plans allocate no transform workspace, and reports actual cuFFT size.
extern "C" int sky_gpu_memory(int nx,int ny,int nz,int ns,uint64_t* data){
  try{CUDA(cudaSetDevice(0));PlanSet p(nx,ny,nz,ns);size_t free=0,total=0;CUDA(cudaMemGetInfo(&free,&total));
    data[0]=p.required();data[1]=p.workspace;data[2]=free;data[3]=total;return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"Sky3D preflight: %s\n",e.what());return 1;}
}
extern "C" void* sky_gpu_create(int nx,int ny,int nz,int ns,const double* spacing,const Z* psi,const double* weights,const int* iq){
  try{CUDA(cudaSetDevice(0));return new State(nx,ny,nz,ns,spacing,psi,weights,iq);}catch(const std::exception& e){failure(e);return nullptr;}
}
extern "C" void sky_gpu_fields(void* pointer,const double* u,const double* b,const double* s,const double* dbm,const double* a,const double* w){
  try{State& p=*(State*)pointer;
    CUDA(cudaMemcpy(p.u,u,size_t(p.g)*2*8,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(p.b,b,size_t(p.g)*2*8,cudaMemcpyHostToDevice));
    for(auto pair:{std::pair<double*,const double*>{p.s,s},{p.dbm,dbm},{p.a,a},{p.w,w}})
      CUDA(cudaMemcpy(pair.first,pair.second,size_t(p.g)*6*8,cudaMemcpyHostToDevice));
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_coulomb_config(void* pointer,const Z* kernel,int periodic,double scale){
  try{State& p=*(State*)pointer;if(p.coul_enabled)return;
    p.periodic=periodic!=0;p.coul_scale=scale;
    CUDA(cudaMemcpy(p.coulkernel,kernel,size_t(p.g)*(p.periodic?1:8)*sizeof(Z),cudaMemcpyHostToDevice));
    p.coul_enabled=true;
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_skyrme(void* pointer,const double* coeff,const double* rho,const double* tau,
  const double* current,const double* spin,const double* so,double* u,double* b,double* s,double* dbm,double* a,double* w){
  try{State& p=*(State*)pointer;Couplings f;std::copy(coeff,coeff+15,f.c);
    for(int q=0;q<2;q++){
      size_t scalar=size_t(p.g)*8,vector=3*scalar;
      CUDA(cudaMemcpyAsync(p.den+p.g*(11*q),rho+p.g*q,scalar,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(1+11*q),tau+p.g*q,scalar,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(2+11*q),current+3*p.g*q,vector,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(5+11*q),spin+3*p.g*q,vector,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(8+11*q),so+3*p.g*q,vector,cudaMemcpyHostToDevice,p.stream));
    }
    p.skyrme(f);CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(u,p.u,size_t(p.g)*2*8,cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(b,p.b,size_t(p.g)*2*8,cudaMemcpyDeviceToHost));
    for(auto pair:{std::pair<double*,double*>{s,p.s},{dbm,p.dbm},{a,p.a},{w,p.w}})
      CUDA(cudaMemcpy(pair.first,pair.second,size_t(p.g)*6*8,cudaMemcpyDeviceToHost));
    p.device_fields=true;
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_coulomb_pull(void* pointer,double* wc){
  try{State& p=*(State*)pointer;CUDA(cudaMemcpy(wc,p.wc,size_t(p.g)*8,cudaMemcpyDeviceToHost));}
  catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_stage(void* pointer,int order,double dt_hbc,int commit,double* density){
  try{State& p=*(State*)pointer;
    if(order<0||!std::isfinite(dt_hbc))throw std::runtime_error("Invalid propagator order or timestep");
    p.stage(order,dt_hbc);CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(density,p.den,size_t(p.g)*22*8,cudaMemcpyDeviceToHost));
    if(commit)std::swap(p.psi,p.result);
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_pull(void* pointer,Z* output){
  try{State& p=*(State*)pointer;CUDA(cudaMemcpy(output,p.psi,size_t(p.count)*sizeof(Z),cudaMemcpyDeviceToHost));}catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_push(void* pointer,const Z* input){
  try{State& p=*(State*)pointer;CUDA(cudaMemcpy(p.psi,input,size_t(p.count)*sizeof(Z),cudaMemcpyHostToDevice));}catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_probe(void* pointer,Z* hout,double* density){
  try{State& p=*(State*)pointer;p.hpsi(p.psi);CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(hout,p.h,size_t(p.count)*sizeof(Z),cudaMemcpyDeviceToHost));p.densities(p.psi);CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(density,p.den,size_t(p.g)*22*8,cudaMemcpyDeviceToHost));
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_destroy(void* pointer){delete (State*)pointer;}
