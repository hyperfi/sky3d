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
#include <utility>
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
template<bool Accumulate=false>
__global__ void unpack_axis(const Z* packed,Z* output,int count,int nx,int ny,int nz,int axis){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=count) return;
  int index=physical_index(t,nx,ny,nz,axis);
  if constexpr(Accumulate) output[index]=output[index]+packed[t];
  else output[index]=packed[t];
}
struct SpectralFactors { double first,second; };

// Each warp handles four 24-point lines. Adjacent lanes access adjacent
// lines, so a strided Y or Z transform still gets coalesced memory accesses.
__device__ inline Z line_shuffle(Z value,int lane){
  return {__shfl_sync(0xffffffffu,value.x,lane),__shfl_sync(0xffffffffu,value.y,lane)};
}

template<bool Inverse>
__device__ inline Z line_rotate(Z value){
  if constexpr(Inverse) return {-value.y,value.x};
  return {value.y,-value.x};
}

// The forward FFT leaves its frequency indices in bit-reversed order.
// Applying the factors in that order saves a shuffle in each transform.
template<int Half,bool Inverse>
__device__ inline Z line_twiddle(Z value,int column){
  if constexpr(Half==2){
    if(column&1) return line_rotate<Inverse>(value);
  }
  if constexpr(Half==4){
    constexpr double half_root_two=0.70710678118654752440;
    int phase=column&3;
    if(phase==1) return half_root_two*(value+line_rotate<Inverse>(value));
    if(phase==2) return line_rotate<Inverse>(value);
    if(phase==3) return half_root_two*(line_rotate<Inverse>(value)-value);
  }
  return value;
}

template<int Half,bool Inverse>
__device__ inline Z line_fft8_step(Z value,int lane){
  int column=lane/4;
  Z other=line_shuffle(value,lane^(4*Half));
  bool upper=(column&Half)!=0;
  if constexpr(Inverse){
    Z left=upper?other:value;
    Z right=line_twiddle<Half,true>(upper?value:other,column);
    return upper?left-right:left+right;
  }
  return upper?line_twiddle<Half,false>(other-value,column):value+other;
}

template<bool Inverse>
__device__ inline Z line_fft8(Z value,int lane){
  if constexpr(Inverse){
    value=line_fft8_step<1,true>(value,lane);
    value=line_fft8_step<2,true>(value,lane);
    return line_fft8_step<4,true>(value,lane);
  }
  value=line_fft8_step<4,false>(value,lane);
  value=line_fft8_step<2,false>(value,lane);
  return line_fft8_step<1,false>(value,lane);
}

template<bool Inverse>
__device__ inline void line_fft3(Z& first,Z& second,Z& third){
  constexpr double half_root_three=0.86602540378443864676;
  Z sum=second+third;
  Z midpoint=first+(-0.5)*sum;
  Z offset=half_root_three*line_rotate<Inverse>(second-third);
  first=first+sum;
  second=midpoint+offset;
  third=midpoint-offset;
}

template<bool Inverse>
__device__ inline void line_fft5(Z (&values)[5]){
  constexpr double c1=0.30901699437494742410,c2=-0.80901699437494742410;
  constexpr double s1=0.95105651629515357212,s2=0.58778525229247312917;
  Z sum14=values[1]+values[4],sum23=values[2]+values[3];
  Z diff14=values[1]-values[4],diff23=values[2]-values[3];
  Z mid1=values[0]+c1*sum14+c2*sum23;
  Z mid2=values[0]+c2*sum14+c1*sum23;
  Z offset1=line_rotate<Inverse>(s1*diff14+s2*diff23);
  Z offset2=line_rotate<Inverse>(s2*diff14-s1*diff23);
  values[0]=values[0]+sum14+sum23;
  values[1]=mid1+offset1;
  values[4]=mid1-offset1;
  values[2]=mid2+offset2;
  values[3]=mid2-offset2;
}

template<int Odd,bool Inverse>
__device__ inline void odd_fft(Z (&values)[Odd]){
  if constexpr(Odd==3) line_fft3<Inverse>(values[0],values[1],values[2]);
  else line_fft5<Inverse>(values);
}

template<int Odd,bool Accumulate,bool Advance=false,bool Contiguous=false,bool Curvature=true>
__global__ void derivative_strided(const Z* input,Z* first,Z* second,int lines,int stride,
                                   const SpectralFactors* factors,Z* term,Z* result,double coefficient){
  constexpr int n=8*Odd;
  int lane=threadIdx.x%32,column=lane/4;
  int line=blockIdx.x*(blockDim.x/8)+(threadIdx.x/32)*4+lane%4;
  // Other grid lengths are even, so all four lines in a participating warp exist.
  if(line>=lines) return;
  // Each warp finishes reading its own lines before writing, including in place.
  int base=Contiguous?line*n:(line/stride)*(n*stride)+line%stride;
  // Good-Thomas indexing for the coprime transform lengths 8 and Odd.
  // The 8-point forward FFT leaves its mode in bit-reversed order.
  int mode_column=((column&1)<<2)|(column&2)|(column>>2);
  constexpr int row_mode=16,column_mode=Odd==3?9:25;
  int points[Odd],modes[Odd];
  Z values[Odd],first_modes[Odd];
  #pragma unroll
  for(int row=0;row<Odd;row++){
    points[row]=base+((8*row+Odd*column)%n)*(Contiguous?1:stride);
    modes[row]=(row_mode*row+column_mode*mode_column)%n;
    values[row]=line_fft8<false>(input[points[row]],lane);
  }
  odd_fft<Odd,false>(values);
  #pragma unroll
  for(int row=0;row<Odd;row++)
    first_modes[row]=factors[modes[row]].first*Z{-values[row].y,values[row].x};
  odd_fft<Odd,true>(first_modes);
  #pragma unroll
  for(int row=0;row<Odd;row++){
    Z value=line_fft8<true>(first_modes[row],lane);
    int point=points[row];
    if constexpr(Advance){
      Z h=first[point]+value;
      Z next{-coefficient*h.y,coefficient*h.x};
      term[point]=next;
      result[point]=result[point]+next;
    }else if constexpr(Accumulate) first[point]=first[point]+value;
    else first[point]=value;
  }
  if(Curvature&&second){
    #pragma unroll
    for(int row=0;row<Odd;row++) values[row]=factors[modes[row]].second*values[row];
    odd_fft<Odd,true>(values);
    #pragma unroll
    for(int row=0;row<Odd;row++)
      second[points[row]]=line_fft8<true>(values[row],lane);
  }
}

// Four lanes per line, with eight adjacent lines sharing a warp.
template<bool Inverse>
__device__ inline Z line_fft4(Z v,int lane){
  constexpr int separation=8;
  if constexpr(Inverse){
    Z peer=line_shuffle(v,lane^separation);
    v=(lane&separation)?peer-v:peer+v;
    peer=line_shuffle(v,lane^(2*separation));
    bool upper=lane&(2*separation);
    Z a=upper?peer:v,b=upper?v:peer;
    if(lane&separation) b=line_rotate<true>(b);
    return upper?a-b:a+b;
  }else{
    Z peer=line_shuffle(v,lane^(2*separation));
    bool upper=lane&(2*separation);
    v=upper?peer-v:peer+v;
    if(upper&&(lane&separation)) v=line_rotate<false>(v);
    peer=line_shuffle(v,lane^separation);
    return (lane&separation)?peer-v:peer+v;
  }
}

template<bool Inverse>
__device__ inline void line_fft9(Z (&values)[9]){
  // For n=r+3*j, transform j, apply W9^(r*k), then transform r.
  #pragma unroll
  for(int r=0;r<3;r++) line_fft3<Inverse>(values[r],values[r+3],values[r+6]);
  constexpr double sign=Inverse?1:-1;
  constexpr Z w1{0.766044443118978035,sign*0.642787609686539326};
  constexpr Z w2{0.173648177666930349,sign*0.984807753012208020};
  constexpr Z w4{-0.939692620785908384,sign*0.342020143325668733};
  values[4]=values[4]*w1;
  values[5]=values[5]*w2;
  values[7]=values[7]*w2;
  values[8]=values[8]*w4;
  #pragma unroll
  for(int k=0;k<3;k++) line_fft3<Inverse>(values[3*k],values[3*k+1],values[3*k+2]);
  Z v1=values[1],v2=values[2],v5=values[5];
  values[1]=values[3]; values[2]=values[6];
  values[3]=v1; values[5]=values[7];
  values[6]=v2; values[7]=v5;
}

template<bool Accumulate,bool Curvature,bool Contiguous=false>
__global__ void derivative36(const Z* input,Z* first,Z* second,int lines,int stride,
                             const SpectralFactors* factors){
  int lane=threadIdx.x%32,column=lane/8;
  int line=blockIdx.x*(blockDim.x/4)+(threadIdx.x/32)*8+lane%8;
  if(line>=lines) return;
  // The other grid lengths are even, so each participating warp owns eight full lines.
  int base=Contiguous?line*36:(line/stride)*(36*stride)+line%stride;
  int frequency_column=((column&1)<<1)|(column>>1);
  // Good-Thomas indexing for the coprime lengths 4 and 9.
  int points[9],modes[9];
  Z values[9],gradients[9];
  #pragma unroll
  for(int r=0;r<9;r++){
    points[r]=base+((4*r+9*column)%36)*(Contiguous?1:stride);
    modes[r]=(28*r+9*frequency_column)%36;
    values[r]=line_fft4<false>(input[points[r]],lane);
  }
  line_fft9<false>(values);
  #pragma unroll
  for(int r=0;r<9;r++){
    gradients[r]=factors[modes[r]].first*Z{-values[r].y,values[r].x};
    if constexpr(Curvature) values[r]=factors[modes[r]].second*values[r];
  }
  line_fft9<true>(gradients);
  #pragma unroll
  for(int r=0;r<9;r++){
    Z v=line_fft4<true>(gradients[r],lane);
    if constexpr(Accumulate) first[points[r]]=first[points[r]]+v;
    else first[points[r]]=v;
  }
  if constexpr(Curvature){
    line_fft9<true>(values);
    #pragma unroll
    for(int r=0;r<9;r++) second[points[r]]=line_fft4<true>(values[r],lane);
  }
}

__global__ void spectral_scale(Z* first,Z* second,int count,int n,const SpectralFactors* factors){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=count) return;
  int k=t%n;
  Z value=first[t];
  if(second) second[t]=factors[k].second*value;
  first[t]=factors[k].first*Z{-value.y,value.x};
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
__device__ inline void apply_h_axis(
    int cell,int spin,int q,int g,int axis,Z value_here,Z value_other,
    Z gradient,Z other_gradient,Z curvature,Z& h,Z& coupled,
    const double* b,const double* dbm,const double* a,const double* w){
  double sign=spin==0?0.5:-0.5,B=b[cell+g*q],D=dbm[field_index(cell,axis,q,g)],A=a[field_index(cell,axis,q,g)];
  double wx=w[field_index(cell,0,q,g)],wy=w[field_index(cell,1,q,g)],wz=w[field_index(cell,2,q,g)];
  // X's coupled derivative is accumulated here, before the Y terms.
  Z value=(axis==1?h+coupled:h)-B*curvature;
  if(axis==0){
    value=value-Z{D,0.5*A-sign*wy}*gradient-(sign*wz)*other_gradient;
    coupled=Z{0,-0.5*(A-2*sign*wy)}*value_here-(sign*wz)*value_other;
  }else if(axis==1){
    value=value-Z{D,0.5*A+sign*wx}*gradient+Z{0,0.5*wz}*other_gradient;
    coupled=Z{0,-0.5*(A+2*sign*wx)}*value_here+Z{0,0.5*wz}*value_other;
  }else{
    value=value-Z{D,0.5*A}*gradient+Z{sign*wx,-0.5*wy}*other_gradient;
    coupled=Z{0,-0.5*A}*value_here+Z{sign*wx,-0.5*wy}*value_other;
  }
  h=value;
}

__global__ void h_axis(const Z* p,const Z* d1,const Z* d2,Z* h,Z* coupled,
                       int count,int g,const int* iq,int axis,const double* b,
                       const double* dbm,const double* a,const double* w){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=count)return;
  int cell=t%g,spin=(t/g)%2,q=iq[t/(2*g)]-1;
  int other=t+(spin==0?g:-g);
  apply_h_axis(cell,spin,q,g,axis,p[t],p[other],d1[t],d1[other],d2[t],
               h[t],coupled[t],b,dbm,a,w);
}
// Both spin components share the same fields and each other's gradient.
template<int Axis>
__global__ void h_axis_spin_pair(const Z* p,const Z* d1,const Z* d2,Z* h,Z* coupled,
                                int g,const int* iq,const double* b,const double* dbm,
                                const double* a,const double* w,const double* u,const double* s){
  int cell=blockIdx.x*blockDim.x+threadIdx.x;
  if(cell>=g) return;
  int q=iq[blockIdx.y]-1;
  int up=2*g*blockIdx.y+cell,down=up+g;
  Z up_p=p[up],down_p=p[down],up_d1=d1[up],down_d1=d1[down];
  Z up_h,down_h,up_c{},down_c{};
  if constexpr(Axis==0){
    int base=cell+g*q;
    double potential=u[base],sx=s[field_index(cell,0,q,g)],sy=s[field_index(cell,1,q,g)],sz=s[field_index(cell,2,q,g)];
    up_h=potential*up_p+Z{sx,-sy}*down_p+sz*up_p;
    down_h=potential*down_p+Z{sx,sy}*up_p+(-sz)*down_p;
  }else{
    up_h=h[up];down_h=h[down];
  }
  if constexpr(Axis==1){
    up_c=coupled[up];down_c=coupled[down];
  }
  apply_h_axis(cell,0,q,g,Axis,up_p,down_p,up_d1,down_d1,d2[up],up_h,up_c,b,dbm,a,w);
  apply_h_axis(cell,1,q,g,Axis,down_p,up_p,down_d1,up_d1,d2[down],down_h,down_c,b,dbm,a,w);
  h[up]=up_h;h[down]=down_h;
  coupled[up]=up_c;coupled[down]=down_c;
}
// Adjacent lanes own the two spin components at the same grid point.
template<int Axis>
__global__ void h_axis24(const Z* input,Z* h,Z* coupled,int count,int g,int stride,
                         const int* iq,const SpectralFactors* factors,const double* b,
                         const double* dbm,const double* a,const double* w){
  int lane=threadIdx.x%32,column=lane/4,spin=lane%2;
  int line=blockIdx.x*(blockDim.x/16)+(threadIdx.x/32)*2+(lane%4)/2;
  if(line>=count/48) return;
  int lines_per_orbital=g/24;
  int orbital=line/lines_per_orbital,spatial_line=line%lines_per_orbital;
  int cell_base=(spatial_line/stride)*(24*stride)+spatial_line%stride;
  int base=(2*orbital+spin)*g;
  int q=iq[orbital]-1;
  int mode_column=((column&1)<<2)|(column&2)|(column>>2);
  int cells[3],modes[3];
  Z points[3],values[3],gradients[3];
  #pragma unroll
  for(int row=0;row<3;row++){
    cells[row]=cell_base+((8*row+3*column)%24)*stride;
    modes[row]=(16*row+9*mode_column)%24;
    points[row]=input[base+cells[row]];
    values[row]=line_fft8<false>(points[row],lane);
  }
  odd_fft<3,false>(values);
  #pragma unroll
  for(int row=0;row<3;row++){
    gradients[row]=factors[modes[row]].first*Z{-values[row].y,values[row].x};
    values[row]=factors[modes[row]].second*values[row];
  }
  odd_fft<3,true>(gradients);
  odd_fft<3,true>(values);
  #pragma unroll
  for(int row=0;row<3;row++){
    int t=base+cells[row];
    Z gradient=line_fft8<true>(gradients[row],lane);
    Z curvature=line_fft8<true>(values[row],lane);
    Z other_gradient=line_shuffle(gradient,lane^1);
    Z other_point=line_shuffle(points[row],lane^1);
    Z accumulated=h[t],mixed;
    if constexpr(Axis==1) mixed=coupled[t];
    apply_h_axis(cells[row],spin,q,g,Axis,points[row],other_point,gradient,other_gradient,
                 curvature,accumulated,mixed,b,dbm,a,w);
    h[t]=accumulated;
    coupled[t]=mixed;
  }
}
__global__ void average_density(double* den,const double* old,int count){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<count)den[t]=0.5*(old[t]+den[t]);
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
#include "reductions.cuh"

struct PlanSet{
  cufftHandle plans[3]={0,0,0};
  cufftHandle realplans[3]={0,0,0},coulplans[2]={0,0};
  cufftHandle fullplan=0;
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
    int dims[3]={z,y,x};size_t fullbytes=0;
    FFT(cufftCreate(&fullplan));FFT(cufftSetAutoAllocation(fullplan,0));
    FFT(cufftMakePlanMany(fullplan,3,dims,nullptr,1,g,nullptr,1,g,CUFFT_Z2Z,2*states,&fullbytes));
    workspace=std::max(workspace,fullbytes);
    }catch(...){for(auto p:plans)if(p)cufftDestroy(p);for(auto p:realplans)if(p)cufftDestroy(p);
      for(auto p:coulplans)if(p)cufftDestroy(p);if(fullplan)cufftDestroy(fullplan);throw;}
  }
  ~PlanSet(){for(auto p:plans)if(p)cufftDestroy(p);for(auto p:realplans)if(p)cufftDestroy(p);
    for(auto p:coulplans)if(p)cufftDestroy(p);if(fullplan)cufftDestroy(fullplan);}
  size_t reduction_bytes()const{return size_t((g+255)/256)*std::max(10*states,38)*8;}
  size_t spectral_factor_bytes()const{return (size_t(nx)+ny+nz)*sizeof(SpectralFactors);}
  bool cache_stage_derivatives()const{return nx==24&&ny==24&&nz==24;}
  uint64_t required()const{return uint64_t(count)*sizeof(Z)*9+uint64_t(g)*1016+uint64_t(states)*52+
    spectral_factor_bytes()+reduction_bytes()+workspace;}
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
using GraphKey=std::tuple<int,double,uintptr_t,int>;
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
  Z* stage_first[3]={},*stage_second[3]={};
  enum class DerivativeCache { none, populate, reuse };
  bool stage_cache_enabled=false,stage_derivatives_valid=false;
  const SpectralFactors* spectral_factors[3];
  double *u,*b,*s,*dbm,*a,*w,*den,*weights,spacing[3];int* iq;
  double *divj,*laprho,*gradrho,*curls,*curlc,*wc;
  Z *coulwork,*coulkernel;
  bool coul_enabled=false,periodic=false,device_fields=false;
  bool density_current=false;
  double coul_scale=0;
  Couplings graph_couplings={};
  bool field_graph_valid=false;
  double *old_den,*statistics,*old_energy;
  void* reduction;
  void* work;
  template<class T>void alloc(T*& p,size_t bytes){CUDA(cudaMalloc(&p,bytes));
    try{allocations.pointers.push_back(p);}catch(...){cudaFree(p);throw;}}
  State(int x,int y,int z,int ns,const double* steps,const Z* p,const double* occ,const int* isospin):PlanSet(x,y,z,ns){
    size_t free=0,total=0;CUDA(cudaMemGetInfo(&free,&total));
    double fraction=0.80;const char* setting=std::getenv("SKY3D_GPU_MEMORY_FRACTION");
    if(setting){char* end=nullptr;fraction=std::strtod(setting,&end);
      if(!end||*end||!std::isfinite(fraction)||fraction<0.05||fraction>0.95)throw std::runtime_error("SKY3D_GPU_MEMORY_FRACTION must be between 0.05 and 0.95");}
    const uint64_t base_required=required();
    const uint64_t cache_bytes=uint64_t(count)*sizeof(Z)*6;
    stage_cache_enabled=cache_stage_derivatives()&&base_required+cache_bytes<=free*fraction;
    const uint64_t total_required=base_required+(stage_cache_enabled?cache_bytes:0);
    std::fprintf(stdout,"GPU memory: required %.1f MiB incl cuFFT workspace; free %.1f MiB, total %.1f MiB\n",total_required/1048576.,free/1048576.,total/1048576.);
    if(base_required>free*fraction)throw std::runtime_error("Insufficient free VRAM with safety headroom. Use SKY3D_BACKEND=cpu. Oversubscription/extra transfers can be slower than parallel CPU; this backend does not page GPU arrays to system RAM.");
    if(total_required>free*0.60)std::fprintf(stderr,"WARNING: limited VRAM headroom; other GPU use may cause allocation failure. Compare a parallel CPU run before assuming a speedup.\n");
    for(int k=0;k<3;k++){if(!std::isfinite(steps[k])||steps[k]<=0)throw std::runtime_error("Grid spacing must be positive");spacing[k]=steps[k];}
    for(int k=0;k<states;k++)if(isospin[k]<1||isospin[k]>2||!std::isfinite(occ[k]))throw std::runtime_error("Invalid occupation or isospin");
    size_t bytes=size_t(count)*sizeof(Z);
    for(Z** p:{&psi,&result,&term,&h,&d1,&d2,&coupled,&frequency,&second})alloc(*p,bytes);
    if(stage_cache_enabled){
      for(int axis=0;axis<3;axis++){
        alloc(stage_first[axis],bytes);
        alloc(stage_second[axis],bytes);
      }
    }
    SpectralFactors* factors;
    alloc(factors,spectral_factor_bytes());
    std::vector<SpectralFactors> host_factors;
    host_factors.reserve(spectral_factor_bytes()/sizeof(SpectralFactors));
    const int lengths[]={nx,ny,nz};
    for(int axis=0;axis<3;axis++){
      int n=lengths[axis];
      spectral_factors[axis]=factors+host_factors.size();
      for(int k=0;k<n;k++){
        int mode=k<n/2?k:k-n;
        double wave=mode*(2.0*PI)/(n*spacing[axis]);
        // First derivative suppresses Nyquist; second derivative retains it.
        host_factors.push_back({k==n/2?0.0:wave/n,-wave*wave/n});
      }
    }
    CUDA(cudaMemcpy(factors,host_factors.data(),spectral_factor_bytes(),cudaMemcpyHostToDevice));
    alloc(u,size_t(g)*2*8);alloc(b,size_t(g)*2*8);
    for(double** p:{&s,&dbm,&a,&w})alloc(*p,size_t(g)*6*8);
    alloc(den,size_t(g)*22*8);alloc(weights,size_t(states)*8);alloc(iq,size_t(states)*4);
    alloc(divj,size_t(g)*2*8);alloc(laprho,size_t(g)*2*8);
    for(double** p:{&gradrho,&curls,&curlc})alloc(*p,size_t(g)*6*8);
    alloc(wc,size_t(g)*8);alloc(coulwork,size_t(g)*8*sizeof(Z));alloc(coulkernel,size_t(g)*8*sizeof(Z));
    alloc(old_den,size_t(g)*22*8);alloc(statistics,size_t(states)*4*8);alloc(old_energy,size_t(states)*8);
    alloc(reduction,reduction_bytes());
    work=nullptr;if(workspace)alloc(work,workspace);
    for(auto plan:plans){FFT(cufftSetWorkArea(plan,work));FFT(cufftSetStream(plan,stream));}
    for(auto plan:realplans){FFT(cufftSetWorkArea(plan,work));FFT(cufftSetStream(plan,stream));}
    for(auto plan:coulplans){FFT(cufftSetWorkArea(plan,work));FFT(cufftSetStream(plan,stream));}
    FFT(cufftSetWorkArea(fullplan,work));FFT(cufftSetStream(fullplan,stream));
    CUDA(cudaMemcpy(psi,p,bytes,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(weights,occ,size_t(states)*8,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(iq,isospin,size_t(states)*4,cudaMemcpyHostToDevice));
    std::fflush(stdout);
  }
  int blocks()const{return (count-1)/256+1;}
  template<int N>std::vector<Values<N>> sums(int groups){
    int nblocks=(g+255)/256;std::vector<Values<N>> raw(nblocks*groups),out(groups);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(raw.data(),reduction,raw.size()*sizeof(Values<N>),cudaMemcpyDeviceToHost));
    for(int group=0;group<groups;group++)for(int block=0;block<nblocks;block++)
      for(int k=0;k<N;k++)out[group].v[k]+=raw[group*nblocks+block].v[k];
    return out;
  }
  void properties(const double* cm,const double* h2m,double* output){
    hpsi(psi);CUDA(cudaMemcpyAsync(term,h,size_t(count)*sizeof(Z),cudaMemcpyDeviceToDevice,stream));
    Z* gradients[3]={d1,d2,coupled};
    for(int axis=0;axis<3;axis++){
      derivative(psi,axis,gradients[axis],h);
      kinetic_add<<<blocks(),256,0,stream>>>(result,h,count,axis==0);
    }
    dim3 grid((g+255)/256,states);
    properties_kernel<<<grid,256,0,stream>>>(psi,term,d1,d2,coupled,result,(Values<10>*)reduction,
      g,nx,ny,nz,spacing[0],spacing[1],spacing[2],cm[0],cm[1],cm[2]);
    auto values=sums<10>(states);std::vector<int> species(states);
    CUDA(cudaMemcpy(species.data(),iq,size_t(states)*4,cudaMemcpyDeviceToHost));
    double volume=spacing[0]*spacing[1]*spacing[2];
    for(int state=0;state<states;state++)for(int k=0;k<10;k++)
      output[k+10*state]=values[state].v[k]*volume*(k==2?h2m[species[state]-1]:1);
  }
  void static_step(const Z* input,const double* energies,double e0,double h2m,double x0,
                   double* spe,double* fluct1,double* fluct2,double* norms,double* delta,Z* output){
    size_t bytes=size_t(count)*sizeof(Z);
    stage_derivatives_valid=false;
    CUDA(cudaMemcpyAsync(psi,input,bytes,cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemcpyAsync(old_energy,energies,size_t(states)*8,cudaMemcpyHostToDevice,stream));
    hpsi(psi);
    static_shift<<<blocks(),256,0,stream>>>(psi,h,term,count,g,old_energy);
    hpsi(term);
    static_statistics<<<dim3((g+255)/256,states),256,0,stream>>>(psi,term,h,(Values<4>*)reduction,g,old_energy);
    auto stats=sums<4>(states);double volume=spacing[0]*spacing[1]*spacing[2];
    for(int state=0;state<states;state++){
      for(int k=0;k<4;k++)stats[state].v[k]*=volume;
      double n=stats[state].v[0],b=stats[state].v[1];
      if(!std::isfinite(n)||n<=0)throw std::runtime_error("Invalid static orbital norm");
      norms[state]=n;fluct1[state]=sqrt(fabs(stats[state].v[2]/n-(b/n)*(b/n)));
      fluct2[state]=sqrt(fabs(stats[state].v[3]/n-(b/n)*(b/n)));
      spe[state]=b+energies[state];delta[state]=(energies[state]-spe[state])/fabs(spe[state]);
    }
    CUDA(cudaMemcpyAsync(statistics,stats.data(),size_t(states)*sizeof(Values<4>),cudaMemcpyHostToDevice,stream));
    if(e0>0){
      static_residual<<<blocks(),256,0,stream>>>(psi,term,count,g,statistics);
      CUDA(cudaMemcpyAsync(result,term,bytes,cudaMemcpyDeviceToDevice,stream));
      FFT(cufftExecZ2Z(fullplan,result,result,CUFFT_FORWARD));
      damp_scale<<<blocks(),256,0,stream>>>(result,count,nx,ny,nz,spacing[0],spacing[1],spacing[2],e0,h2m);
      FFT(cufftExecZ2Z(fullplan,result,result,CUFFT_INVERSE));
    }
    static_update<<<blocks(),256,0,stream>>>(psi,result,term,statistics,count,g,x0,e0>0);
    CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(output,psi,bytes,cudaMemcpyDeviceToHost));density_current=false;
  }
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
  void enqueue_skyrme(Couplings f){
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
  void skyrme(Couplings f){
    const char* setting=std::getenv("SKY3D_GPU_GRAPHS");
    if(setting&&std::string(setting)=="0"){enqueue_skyrme(f);return;}
    if(setting&&std::string(setting)!="1")throw std::runtime_error("SKY3D_GPU_GRAPHS must be 0 or 1");
    GraphKey key{-1,0,0,0};auto found=graphs.entries.find(key);
    bool same=field_graph_valid&&std::equal(f.c,f.c+15,graph_couplings.c);
    if(found!=graphs.entries.end()&&!same){cudaGraphExecDestroy(found->second);graphs.entries.erase(found);found=graphs.entries.end();}
    if(found==graphs.entries.end()){
      cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
      CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      try{enqueue_skyrme(f);}catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
      CUDA(cudaStreamEndCapture(stream,&graph));
      try{CUDA(cudaGraphInstantiateWithFlags(&executable,graph,0));}catch(...){cudaGraphDestroy(graph);throw;}
      cudaGraphDestroy(graph);
      try{found=graphs.entries.emplace(key,executable).first;}catch(...){cudaGraphExecDestroy(executable);throw;}
      graph_couplings=f;field_graph_valid=true;
    }
    CUDA(cudaGraphLaunch(found->second,stream));
  }
  template<bool Accumulate=false>
  void derivative(const Z* input,int axis,Z* first,Z* second_output=nullptr){
    int n=axis==0?nx:(axis==1?ny:nz);
    if(!Accumulate&&axis==0&&n==24){
      int lines=count/24;
      if(second_output)
        derivative_strided<3,false,false,true,true><<<(lines+31)/32,256,0,stream>>>
          (input,first,second_output,lines,1,spectral_factors[axis],nullptr,nullptr,0);
      else
        derivative_strided<3,false,false,true,false><<<(lines+31)/32,256,0,stream>>>
          (input,first,nullptr,lines,1,spectral_factors[axis],nullptr,nullptr,0);
      return;
    }
    if(!Accumulate&&axis==0&&(n==36||n==40)){
      int lines=count/n;
      if(n==36){
        if(second_output)
          derivative36<false,true,true><<<(lines+31)/32,128,0,stream>>>
            (input,first,second_output,lines,1,spectral_factors[axis]);
        else
          derivative36<false,false,true><<<(lines+31)/32,128,0,stream>>>
            (input,first,nullptr,lines,1,spectral_factors[axis]);
      }else{
        if(second_output)
          derivative_strided<5,false,false,true,true><<<(lines+31)/32,256,0,stream>>>
            (input,first,second_output,lines,1,spectral_factors[axis],nullptr,nullptr,0);
        else
          derivative_strided<5,false,false,true,false><<<(lines+31)/32,256,0,stream>>>
            (input,first,nullptr,lines,1,spectral_factors[axis],nullptr,nullptr,0);
      }
      return;
    }
    if(axis!=0&&n==36){
      int lines=count/n,stride=axis==1?nx:nx*ny;
      if(second_output)
        derivative36<Accumulate,true><<<(lines+31)/32,128,0,stream>>>(input,first,second_output,lines,stride,spectral_factors[axis]);
      else
        derivative36<Accumulate,false><<<(lines+31)/32,128,0,stream>>>(input,first,nullptr,lines,stride,spectral_factors[axis]);
      return;
    }
    if(axis!=0&&(n==24||n==40)){
      int lines=count/n,stride=axis==1?nx:nx*ny;
      if(n==24)
        derivative_strided<3,Accumulate><<<(lines+31)/32,256,0,stream>>>(input,first,second_output,lines,stride,spectral_factors[axis],nullptr,nullptr,0);
      else
        derivative_strided<5,Accumulate><<<(lines+31)/32,256,0,stream>>>(input,first,second_output,lines,stride,spectral_factors[axis],nullptr,nullptr,0);
      return;
    }
    // X lines are contiguous. cuFFT's out-of-place input is read-only,
    // though its execution API takes a non-const pointer.
    const bool contiguous=axis==0;
    if(!contiguous)
      pack_axis<<<blocks(),256,0,stream>>>(input,frequency,count,nx,ny,nz,axis);
    Z* fft_input=contiguous?const_cast<Z*>(input):frequency;
    FFT(cufftExecZ2Z(plans[axis],fft_input,frequency,CUFFT_FORWARD));
    spectral_scale<<<blocks(),256,0,stream>>>(frequency,second_output?second:nullptr,count,n,spectral_factors[axis]);
    if(second_output){
      FFT(cufftExecZ2Z(plans[axis],second,contiguous?second_output:second,CUFFT_INVERSE));
      if(!contiguous) unpack_axis<<<blocks(),256,0,stream>>>(second,second_output,count,nx,ny,nz,axis);
    }
    FFT(cufftExecZ2Z(plans[axis],frequency,contiguous?first:frequency,CUFFT_INVERSE));
    if(!contiguous) unpack_axis<Accumulate><<<blocks(),256,0,stream>>>(frequency,first,count,nx,ny,nz,axis);
  }
  void apply_axis(const Z* input,const Z* first,const Z* second,int axis,bool pair_spins){
    if(!pair_spins){
      h_axis<<<blocks(),256,0,stream>>>(input,first,second,h,coupled,count,g,iq,axis,b,dbm,a,w);
      return;
    }
    const dim3 grid((g-1)/256+1,states);
    if(axis==0)
      h_axis_spin_pair<0><<<grid,256,0,stream>>>(input,first,second,h,coupled,g,iq,b,dbm,a,w,u,s);
    else if(axis==1)
      h_axis_spin_pair<1><<<grid,256,0,stream>>>(input,first,second,h,coupled,g,iq,b,dbm,a,w,u,s);
    else
      h_axis_spin_pair<2><<<grid,256,0,stream>>>(input,first,second,h,coupled,g,iq,b,dbm,a,w,u,s);
  }
  void hpsi(const Z* input,DerivativeCache cache=DerivativeCache::none,
            bool advance=false,double coefficient=0){
    const bool fuse_axes=nx==24&&ny==24&&nz==24;
    const bool pair_spins=nx==ny&&ny==nz&&(nx==36||nx==40)&&states<=65535;
    // On these meshes the paired X update also initializes h.
    if(!pair_spins) h_local<<<blocks(),256,0,stream>>>(input,h,count,g,iq,u,s);
    for(int axis=0;axis<3;axis++){
      if(cache!=DerivativeCache::none){
        Z* first=stage_first[axis];
        Z* second=stage_second[axis];
        if(cache==DerivativeCache::populate) derivative(input,axis,first,second);
        apply_axis(input,first,second,axis,pair_spins);
      }else if(fuse_axes&&axis==1)
        h_axis24<1><<<(count/48+15)/16,256,0,stream>>>(input,h,coupled,count,g,nx,iq,spectral_factors[1],b,dbm,a,w);
      else if(fuse_axes&&axis==2)
        h_axis24<2><<<(count/48+15)/16,256,0,stream>>>(input,h,coupled,count,g,nx*ny,iq,spectral_factors[2],b,dbm,a,w);
      else{
        derivative(input,axis,d1,d2);
        apply_axis(input,d1,d2,axis,pair_spins);
      }
      // The X inverse can reuse coupled because the forward FFT has
      // consumed it. Y and Z add to h as they store the inverse result.
      if(axis==0) derivative(coupled,axis,coupled);
      else if(axis==2&&advance){
        const int lines=count/24;
        derivative_strided<3,true,true><<<(lines+31)/32,256,0,stream>>>
          (coupled,h,nullptr,lines,nx*ny,spectral_factors[axis],term,result,coefficient);
      }else derivative<true>(coupled,axis,h);
    }
  }
  void densities(const Z* input){
    derivative(input,0,d1);derivative(input,1,d2);derivative(input,2,coupled);
    density_kernel<<<(2*g-1)/256+1,256,0,stream>>>(input,d1,d2,coupled,den,g,states,iq,weights);
  }
  void enqueue_stage(int order,double dt_hbc,DerivativeCache cache){
    size_t bytes=size_t(count)*sizeof(Z);
    CUDA(cudaMemcpyAsync(result,psi,bytes,cudaMemcpyDeviceToDevice,stream));
    const bool advance=nx==24&&ny==24&&nz==24;
    // The last derivative and the recurrence write the same grid points.
    if(order>0){
      hpsi(psi,cache,advance,-dt_hbc);
      if(!advance) recurrence<<<blocks(),256,0,stream>>>(term,result,h,count,-dt_hbc);
    }
    for(int m=2;m<=order;m++){
      const double coefficient=-dt_hbc/m;
      hpsi(term,DerivativeCache::none,advance,coefficient);
      if(!advance) recurrence<<<blocks(),256,0,stream>>>(term,result,h,count,coefficient);
    }
    densities(result);CUDA(cudaGetLastError());
  }
  void stage(int order,double dt_hbc,bool commit){
    // Fields change between stages, but the starting orbital derivatives do not.
    DerivativeCache cache=DerivativeCache::none;
    if(order>0&&stage_cache_enabled){
      if(!commit) cache=DerivativeCache::populate;
      else if(stage_derivatives_valid) cache=DerivativeCache::reuse;
    }
    stage_derivatives_valid=cache==DerivativeCache::populate;
    const char* setting=std::getenv("SKY3D_GPU_GRAPHS");
    if(setting&&std::string(setting)=="0"){enqueue_stage(order,dt_hbc,cache);return;}
    if(setting&&std::string(setting)!="1")throw std::runtime_error("SKY3D_GPU_GRAPHS must be 0 or 1");
    // Each alternating wavefunction buffer and Taylor order has its own graph.
    // Field values change in place; timestep and cache use select different keys.
    GraphKey key{order,dt_hbc,reinterpret_cast<uintptr_t>(psi),static_cast<int>(cache)};
    auto found=graphs.entries.find(key);
    if(found==graphs.entries.end()){
      cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
      CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      try{enqueue_stage(order,dt_hbc,cache);}catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
      CUDA(cudaStreamEndCapture(stream,&graph));
      try{CUDA(cudaGraphInstantiateWithFlags(&executable,graph,0));}catch(...){cudaGraphDestroy(graph);throw;}
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
    CUDA(cudaMemcpy(p.u,u,size_t(p.g)*2*8,cudaMemcpyHostToDevice));
    if(p.device_fields)return;
    CUDA(cudaMemcpy(p.b,b,size_t(p.g)*2*8,cudaMemcpyHostToDevice));
    const std::pair<double*,const double*> fields[]={{p.s,s},{p.dbm,dbm},{p.a,a},{p.w,w}};
    for(const auto& [destination,source]:fields)
      CUDA(cudaMemcpy(destination,source,size_t(p.g)*6*8,cudaMemcpyHostToDevice));
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_coulomb_config(void* pointer,const Z* kernel,int periodic,double scale){
  try{State& p=*(State*)pointer;if(p.coul_enabled)return;
    p.periodic=periodic!=0;p.coul_scale=scale;
    CUDA(cudaMemcpy(p.coulkernel,kernel,size_t(p.g)*(p.periodic?1:8)*sizeof(Z),cudaMemcpyHostToDevice));
    p.coul_enabled=true;p.field_graph_valid=false;
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_skyrme(void* pointer,const double* coeff,const double* rho,const double* tau,
  const double* current,const double* spin,const double* so,double* u,double* b,double* s,double* dbm,double* a,double* w,int resident){
  try{State& p=*(State*)pointer;Couplings f;std::copy(coeff,coeff+15,f.c);
    if(!resident||!p.density_current)for(int q=0;q<2;q++){
      size_t scalar=size_t(p.g)*8,vector=3*scalar;
      CUDA(cudaMemcpyAsync(p.den+p.g*(11*q),rho+p.g*q,scalar,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(1+11*q),tau+p.g*q,scalar,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(2+11*q),current+3*p.g*q,vector,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(5+11*q),spin+3*p.g*q,vector,cudaMemcpyHostToDevice,p.stream));
      CUDA(cudaMemcpyAsync(p.den+p.g*(8+11*q),so+3*p.g*q,vector,cudaMemcpyHostToDevice,p.stream));
    }
    p.skyrme(f);CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(u,p.u,size_t(p.g)*2*8,cudaMemcpyDeviceToHost));
    if(!resident){
      CUDA(cudaMemcpy(b,p.b,size_t(p.g)*2*8,cudaMemcpyDeviceToHost));
      const std::pair<double*,const double*> fields[]={{s,p.s},{dbm,p.dbm},{a,p.a},{w,p.w}};
      for(const auto& [destination,source]:fields)
        CUDA(cudaMemcpy(destination,source,size_t(p.g)*6*8,cudaMemcpyDeviceToHost));
    }
    p.device_fields=true;p.density_current=true;
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_coulomb_pull(void* pointer,double* wc){
  try{State& p=*(State*)pointer;CUDA(cudaMemcpy(wc,p.wc,size_t(p.g)*8,cudaMemcpyDeviceToHost));}
  catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_stage(void* pointer,int order,double dt_hbc,int commit,double* density,int resident){
  try{State& p=*(State*)pointer;
    if(order<0||!std::isfinite(dt_hbc))throw std::runtime_error("Invalid propagator order or timestep");
    if(resident&&!commit)CUDA(cudaMemcpyAsync(p.old_den,p.den,size_t(p.g)*22*8,cudaMemcpyDeviceToDevice,p.stream));
    p.stage(order,dt_hbc,commit!=0);
    if(resident&&!commit)average_density<<<(22*p.g-1)/256+1,256,0,p.stream>>>(p.den,p.old_den,22*p.g);
    CUDA(cudaStreamSynchronize(p.stream));
    if(!resident)CUDA(cudaMemcpy(density,p.den,size_t(p.g)*22*8,cudaMemcpyDeviceToHost));
    p.density_current=true;
    if(commit)std::swap(p.psi,p.result);
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_pull(void* pointer,Z* output){
  try{State& p=*(State*)pointer;CUDA(cudaStreamSynchronize(p.stream));CUDA(cudaMemcpy(output,p.psi,size_t(p.count)*sizeof(Z),cudaMemcpyDeviceToHost));}catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_push(void* pointer,const Z* input){
  try{State& p=*(State*)pointer;CUDA(cudaMemcpy(p.psi,input,size_t(p.count)*sizeof(Z),cudaMemcpyHostToDevice));p.density_current=false;p.stage_derivatives_valid=false;}catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_fields_pull(void* pointer,double* u,double* b,double* s,double* dbm,double* a,double* w){
  try{State& p=*(State*)pointer;CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(u,p.u,size_t(p.g)*2*8,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(b,p.b,size_t(p.g)*2*8,cudaMemcpyDeviceToHost));
    const std::pair<double*,const double*> fields[]={{s,p.s},{dbm,p.dbm},{a,p.a},{w,p.w}};
    for(const auto& [destination,source]:fields)
      CUDA(cudaMemcpy(destination,source,size_t(p.g)*6*8,cudaMemcpyDeviceToHost));
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_density_pull(void* pointer,double* density){
  try{State& p=*(State*)pointer;CUDA(cudaStreamSynchronize(p.stream));CUDA(cudaMemcpy(density,p.den,size_t(p.g)*22*8,cudaMemcpyDeviceToHost));}
  catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_density_refresh(void* pointer,const Z* psi,const double* weights,double* density){
  try{State& p=*(State*)pointer;
    p.stage_derivatives_valid=false;
    CUDA(cudaMemcpyAsync(p.psi,psi,size_t(p.count)*sizeof(Z),cudaMemcpyHostToDevice,p.stream));
    CUDA(cudaMemcpyAsync(p.weights,weights,size_t(p.states)*8,cudaMemcpyHostToDevice,p.stream));
    p.densities(p.psi);CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(density,p.den,size_t(p.g)*22*8,cudaMemcpyDeviceToHost));p.density_current=true;
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_properties(void* pointer,const double* cm,const double* h2m,double* values){
  try{((State*)pointer)->properties(cm,h2m,values);}catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_moments_first(void* pointer,double* values){
  try{State& p=*(State*)pointer;
    moments_first<<<dim3((p.g+255)/256,2),256,0,p.stream>>>(p.den,(Values<19>*)p.reduction,
      p.g,p.nx,p.ny,p.nz,p.spacing[0],p.spacing[1],p.spacing[2]);
    auto result=p.sums<19>(2);for(int q=0;q<2;q++)for(int k=0;k<7;k++)values[k+7*q]=result[q].v[k];
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_moments_second(void* pointer,const double* cm,double* values){
  try{State& p=*(State*)pointer;
    moments_second<<<dim3((p.g+255)/256,2),256,0,p.stream>>>(p.den,(Values<19>*)p.reduction,
      p.g,p.nx,p.ny,p.nz,p.spacing[0],p.spacing[1],p.spacing[2],cm[0],cm[1],cm[2],cm[3],cm[4],cm[5]);
    auto result=p.sums<19>(2);for(int q=0;q<2;q++)for(int k=0;k<19;k++)values[k+19*q]=result[q].v[k];
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_static_step(void* pointer,const Z* input,const double* energies,double e0,double h2m,double x0,
  double* spe,double* fluct1,double* fluct2,double* norms,double* delta,Z* output){
  try{((State*)pointer)->static_step(input,energies,e0,h2m,x0,spe,fluct1,fluct2,norms,delta,output);}
  catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_hamiltonian(void* pointer,const Z* input,Z* output){
  try{State& p=*(State*)pointer;p.stage_derivatives_valid=false;
    CUDA(cudaMemcpyAsync(p.psi,input,size_t(p.count)*sizeof(Z),cudaMemcpyHostToDevice,p.stream));
    p.hpsi(p.psi);CUDA(cudaStreamSynchronize(p.stream));CUDA(cudaMemcpy(output,p.h,size_t(p.count)*sizeof(Z),cudaMemcpyDeviceToHost));}
  catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_probe(void* pointer,Z* hout,double* density){
  try{State& p=*(State*)pointer;p.hpsi(p.psi);CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(hout,p.h,size_t(p.count)*sizeof(Z),cudaMemcpyDeviceToHost));p.densities(p.psi);CUDA(cudaStreamSynchronize(p.stream));
    CUDA(cudaMemcpy(density,p.den,size_t(p.g)*22*8,cudaMemcpyDeviceToHost));
  }catch(const std::exception& e){failure(e);}
}
extern "C" void sky_gpu_destroy(void* pointer){delete (State*)pointer;}
