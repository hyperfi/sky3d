// Ordered block reductions in FP64; no global floating-point atomics.
#include <cub/block/block_reduce.cuh>
template<int N> struct Values {double v[N];};
template<int N> struct SumValues {
  __device__ Values<N> operator()(Values<N> a,const Values<N>& b)const{
    for(int k=0;k<N;k++)a.v[k]+=b.v[k];return a;
  }
};
__device__ double coordinate(int i,int n,double d){return (i+0.5-n/2)*d;}
__device__ double legendre0(int l,double x){
  if(l==0)return 1;double a=1,b=x;
  for(int k=2;k<=l;k++){double c=((2*k-1)*x*b-(k-1)*a)/k;a=b;b=c;}return b;
}
__global__ void moments_first(const double* den,Values<19>* partial,int g,int nx,int ny,int nz,
                              double dx,double dy,double dz){
  using Reduce=cub::BlockReduce<Values<19>,256>;
  __shared__ typename Reduce::TempStorage temp;
  int cell=blockIdx.x*blockDim.x+threadIdx.x,q=blockIdx.y;Values<19> v={};
  if(cell<g){
    double volume=dx*dy*dz,rho=den[cell+11*g*q],weight=volume*rho;
    v.v[0]=weight;
    v.v[1]=weight*coordinate(cell%nx,nx,dx);
    v.v[2]=weight*coordinate((cell/nx)%ny,ny,dy);
    v.v[3]=weight*coordinate(cell/(nx*ny),nz,dz);
    for(int k=0;k<3;k++)v.v[4+k]=volume*den[cell+g*(2+k+11*q)];
  }
  auto value=Reduce(temp).Reduce(v,SumValues<19>());
  if(threadIdx.x==0)partial[blockIdx.x+gridDim.x*q]=value;
}
__global__ void moments_second(const double* den,Values<19>* partial,int g,int nx,int ny,int nz,
  double dx,double dy,double dz,double nxcm,double nycm,double nzcm,double pxcm,double pycm,double pzcm){
  using Reduce=cub::BlockReduce<Values<19>,256>;
  __shared__ typename Reduce::TempStorage temp;
  int cell=blockIdx.x*blockDim.x+threadIdx.x,q=blockIdx.y;Values<19> v={};
  if(cell<g){
    double x0=coordinate(cell%nx,nx,dx),y0=coordinate((cell/nx)%ny,ny,dy),z0=coordinate(cell/(nx*ny),nz,dz);
    double x=x0-(q==0?nxcm:pxcm),y=y0-(q==0?nycm:pycm),z=z0-(q==0?nzcm:pzcm);
    double xx=x*x,yy=y*y,zz=z*z,r2=xx+yy+zz,r=sqrt(r2),vol=dx*dy*dz*den[cell+11*g*q];
    v.v[0]=vol*r2;v.v[1]=vol*pow(r2,1.5);v.v[2]=vol*r2*r2;
    v.v[3]=vol*(xx+xx-yy-zz);v.v[4]=3*vol*x*y;v.v[5]=3*vol*x*z;
    v.v[6]=vol*(yy+yy-xx-zz);v.v[7]=3*vol*y*z;v.v[8]=vol*(zz+zz-xx-yy);
    v.v[9]=vol*xx;v.v[10]=vol*yy;v.v[11]=vol*zz;
    v.v[12]=vol*0.5*sqrt(1/PI)*r2;
    double r0=sqrt(x0*x0+y0*y0+z0*z0),y10=sqrt(3/(4*PI))*z0/r0;
    // Preserve the existing native dipole correction, including its vol factor.
    double eta=vol*r2*5/3;
    v.v[13]=vol*y10*(r0*r0*r0-r0*eta)*sqrt(3.0);
    v.v[14]=vol*y10*r0*sqrt(3.0);
    for(int l=2;l<=5;l++){
      double harmonic=sqrt((2*l+1)/(4*PI))*legendre0(l,z/r);
      v.v[13+l]=vol*harmonic*pow(r,double(l))*sqrt(2*l+1.0);
    }
  }
  auto value=Reduce(temp).Reduce(v,SumValues<19>());
  if(threadIdx.x==0)partial[blockIdx.x+gridDim.x*q]=value;
}
__global__ void properties_kernel(const Z* psi,const Z* h,const Z* gx,const Z* gy,const Z* gz,
  const Z* lap,Values<10>* partial,int g,int nx,int ny,int nz,double dx,double dy,double dz,
  double cmx,double cmy,double cmz){
  using Reduce=cub::BlockReduce<Values<10>,256>;
  __shared__ typename Reduce::TempStorage temp;
  int cell=blockIdx.x*blockDim.x+threadIdx.x,state=blockIdx.y;Values<10> v={};
  if(cell<g){
    int base=cell+state*2*g;
    double x=coordinate(cell%nx,nx,dx)-cmx,y=coordinate((cell/nx)%ny,ny,dy)-cmy,
           z=coordinate(cell/(nx*ny),nz,dz)-cmz;
    int reverse=g-1-cell;
    for(int spin=0;spin<2;spin++){
      int i=base+spin*g;Z p=psi[i],px=conjugate(p)*gx[i],py=conjugate(p)*gy[i],pz=conjugate(p)*gz[i];
      v.v[0]+=norm2(p);v.v[1]+=(conjugate(p)*h[i]).x;
      v.v[2]-=(conjugate(p)*lap[i]).x;
      v.v[3]+=(conjugate(p)*psi[reverse+state*2*g+spin*g]).x;
      v.v[4]+=y*pz.y-z*py.y;v.v[5]+=z*px.y-x*pz.y;v.v[6]+=x*py.y-y*px.y;
    }
    Z cross=conjugate(psi[base])*psi[base+g];
    v.v[7]=cross.x;v.v[8]=cross.y;v.v[9]=0.5*(norm2(psi[base])-norm2(psi[base+g]));
  }
  auto value=Reduce(temp).Reduce(v,SumValues<10>());
  if(threadIdx.x==0)partial[blockIdx.x+gridDim.x*state]=value;
}
__global__ void kinetic_add(Z* lap,const Z* d2,int count,bool first){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<count)lap[t]=first?d2[t]:lap[t]+d2[t];
}
__global__ void static_shift(const Z* psi,const Z* h,Z* residual,int count,int g,const double* energy){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<count)residual[t]=h[t]-energy[t/(2*g)]*psi[t];
}
__global__ void static_statistics(const Z* psi,const Z* r,const Z* h2,Values<4>* partial,int g,const double* energy){
  using Reduce=cub::BlockReduce<Values<4>,256>;
  __shared__ typename Reduce::TempStorage temp;
  int cell=blockIdx.x*blockDim.x+threadIdx.x,state=blockIdx.y;Values<4> v={};
  if(cell<g)for(int spin=0;spin<2;spin++){
    int i=cell+(2*state+spin)*g;Z p=psi[i],a=r[i],b=h2[i]-energy[state]*a;
    v.v[0]+=norm2(p);v.v[1]+=(conjugate(p)*a).x;
    v.v[2]+=(conjugate(p)*b).x;v.v[3]+=norm2(a);
  }
  auto value=Reduce(temp).Reduce(v,SumValues<4>());
  if(threadIdx.x==0)partial[blockIdx.x+gridDim.x*state]=value;
}
__global__ void static_residual(const Z* psi,Z* r,int count,int g,const double* stats){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<count)r[t]=r[t]-stats[4*(t/(2*g))+1]*psi[t];
}
__global__ void damp_scale(Z* data,int count,int nx,int ny,int nz,double dx,double dy,double dz,double e0,double h2m){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;
  int g=nx*ny*nz,cell=t%g,x=cell%nx,y=(cell/nx)%ny,z=cell/(nx*ny);
  int kx=x<nx/2?x:x-nx,ky=y<ny/2?y:y-ny,kz=z<nz/2?z:z-nz;
  double ax=kx*2*PI/(nx*dx),ay=ky*2*PI/(ny*dy),az=kz*2*PI/(nz*dz);
  data[t]=(1/(g*(e0+h2m*(ax*ax+ay*ay+az*az))))*data[t];
}
__global__ void static_update(Z* psi,const Z* step,const Z* residual,const double* stats,int count,int g,double x0,bool damped){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;
  psi[t]=damped?psi[t]-x0*step[t]:(1+x0*stats[4*(t/(2*g))+1])*psi[t]-x0*residual[t];
}
