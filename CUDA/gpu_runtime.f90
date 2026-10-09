! GPU implementation; build this module instead of Code/gpu_runtime.f90.
MODULE GPU_Runtime
  USE Params, ONLY: db
  USE ISO_C_BINDING
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: gpu_enabled,gpu_initialize,gpu_fields,gpu_stage,gpu_pull,gpu_push,gpu_probe,gpu_finish
  PUBLIC :: gpu_fields_enabled,gpu_skyrme,gpu_coulomb_config,gpu_coulomb_pull
  LOGICAL :: gpu_enabled=.FALSE.
  LOGICAL :: gpu_fields_enabled=.TRUE.
  TYPE(C_PTR) :: handle=C_NULL_PTR
  REAL(db),ALLOCATABLE,TARGET :: density(:,:,:,:,:)
  INTERFACE
    SUBROUTINE c_coul_config(p,q,periodic,scale) BIND(C,NAME='sky_gpu_coulomb_config')
      IMPORT
      TYPE(C_PTR),VALUE :: p,q
      INTEGER(C_INT),VALUE :: periodic
      REAL(C_DOUBLE),VALUE :: scale
    END SUBROUTINE
    SUBROUTINE c_coul_pull(p,wc) BIND(C,NAME='sky_gpu_coulomb_pull')
      IMPORT
      TYPE(C_PTR),VALUE :: p,wc
    END SUBROUTINE
    SUBROUTINE c_skyrme(p,f,r,t,c,sd,j,u,b,s,dbm,a,w) BIND(C,NAME='sky_gpu_skyrme')
      IMPORT
      TYPE(C_PTR),VALUE :: p,f,r,t,c,sd,j,u,b,s,dbm,a,w
    END SUBROUTINE
    FUNCTION c_create(nx,ny,nz,ns,spacing,psi,weights,iq) BIND(C,NAME='sky_gpu_create') RESULT(p)
      IMPORT
      INTEGER(C_INT),VALUE :: nx,ny,nz,ns
      TYPE(C_PTR),VALUE :: spacing,psi,weights,iq
      TYPE(C_PTR) :: p
    END FUNCTION
    SUBROUTINE c_fields(p,u,b,s,dbm,a,w) BIND(C,NAME='sky_gpu_fields')
      IMPORT
      TYPE(C_PTR),VALUE :: p,u,b,s,dbm,a,w
    END SUBROUTINE
    SUBROUTINE c_stage(p,order,dt_hbc,commit,den) BIND(C,NAME='sky_gpu_stage')
      IMPORT
      TYPE(C_PTR),VALUE :: p,den
      INTEGER(C_INT),VALUE :: order,commit
      REAL(C_DOUBLE),VALUE :: dt_hbc
    END SUBROUTINE
    SUBROUTINE c_pull(p,ps) BIND(C,NAME='sky_gpu_pull')
      IMPORT
      TYPE(C_PTR),VALUE :: p,ps
    END SUBROUTINE
    SUBROUTINE c_push(p,ps) BIND(C,NAME='sky_gpu_push')
      IMPORT
      TYPE(C_PTR),VALUE :: p,ps
    END SUBROUTINE
    SUBROUTINE c_probe(p,h,den) BIND(C,NAME='sky_gpu_probe')
      IMPORT
      TYPE(C_PTR),VALUE :: p,h,den
    END SUBROUTINE
    SUBROUTINE c_finish(p) BIND(C,NAME='sky_gpu_destroy')
      IMPORT
      TYPE(C_PTR),VALUE :: p
    END SUBROUTINE
  END INTERFACE
CONTAINS
  SUBROUTINE gpu_initialize(ps,weights,iq,spacing,tfft,tmpi)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(INOUT) :: ps(:,:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(IN) :: weights(:)
    REAL(db),TARGET,INTENT(IN) :: spacing(3)
    INTEGER,CONTIGUOUS,TARGET,INTENT(IN) :: iq(:)
    LOGICAL,INTENT(IN) :: tfft,tmpi
    CHARACTER(32) :: backend
    INTEGER :: status
    CALL get_environment_variable('SKY3D_BACKEND',backend,STATUS=status)
    IF(status==1) backend='gpu'
    IF(TRIM(backend)=='cpu') RETURN
    IF(TRIM(backend)/='gpu') ERROR STOP 'SKY3D_BACKEND must be cpu or gpu'
    IF(tmpi) ERROR STOP 'GPU pilot supports one process; MPI GPU mode is not implemented'
    IF(.NOT.tfft) ERROR STOP 'GPU pilot requires tfft=T'
    CALL get_environment_variable('SKY3D_GPU_FIELDS',backend,STATUS=status)
    gpu_fields_enabled=.TRUE.
    IF(status==0) THEN
      IF(TRIM(backend)/='0'.AND.TRIM(backend)/='1') ERROR STOP 'SKY3D_GPU_FIELDS must be 0 or 1'
      gpu_fields_enabled=TRIM(backend)=='1'
    ENDIF
    IF(db/=C_DOUBLE.OR.STORAGE_SIZE(ps)/=128) ERROR STOP 'GPU backend requires complex FP64'
    handle=c_create(SIZE(ps,1),SIZE(ps,2),SIZE(ps,3),SIZE(ps,5), &
                    C_LOC(spacing),C_LOC(ps),C_LOC(weights),C_LOC(iq))
    IF(.NOT.C_ASSOCIATED(handle)) ERROR STOP 'GPU initialization failed'
    ALLOCATE(density(SIZE(ps,1),SIZE(ps,2),SIZE(ps,3),11,2))
    gpu_enabled=.TRUE.
  END SUBROUTINE
  SUBROUTINE gpu_coulomb_config(q,periodic,scale)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(IN) :: q(:,:,:)
    LOGICAL,INTENT(IN) :: periodic
    REAL(db),INTENT(IN) :: scale
    CALL c_coul_config(handle,C_LOC(q),MERGE(1,0,periodic),scale)
  END SUBROUTINE
  SUBROUTINE gpu_coulomb_pull(wc)
    REAL(db),CONTIGUOUS,TARGET,INTENT(OUT) :: wc(:,:,:)
    CALL c_coul_pull(handle,C_LOC(wc))
  END SUBROUTINE
  SUBROUTINE gpu_skyrme(f,r,t,c,sd,j,u,b,s,dbm,a,w)
    REAL(db),CONTIGUOUS,TARGET,INTENT(IN) :: f(:),r(:,:,:,:),t(:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(IN) :: c(:,:,:,:,:),sd(:,:,:,:,:),j(:,:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(OUT) :: u(:,:,:,:),b(:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(OUT) :: s(:,:,:,:,:),dbm(:,:,:,:,:),a(:,:,:,:,:),w(:,:,:,:,:)
    CALL c_skyrme(handle,C_LOC(f),C_LOC(r),C_LOC(t),C_LOC(c),C_LOC(sd),C_LOC(j), &
                  C_LOC(u),C_LOC(b),C_LOC(s),C_LOC(dbm),C_LOC(a),C_LOC(w))
  END SUBROUTINE
  SUBROUTINE gpu_fields(u,b,s,dbm,a,w)
    REAL(db),CONTIGUOUS,TARGET,INTENT(IN) :: u(:,:,:,:),b(:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(IN) :: s(:,:,:,:,:),dbm(:,:,:,:,:),a(:,:,:,:,:),w(:,:,:,:,:)
    IF(gpu_enabled) CALL c_fields(handle,C_LOC(u),C_LOC(b),C_LOC(s),C_LOC(dbm),C_LOC(a),C_LOC(w))
  END SUBROUTINE
  SUBROUTINE gpu_stage(order,dt,hbc,commit,r,t,c,s,j)
    INTEGER,INTENT(IN) :: order
    REAL(db),INTENT(IN) :: dt,hbc
    LOGICAL,INTENT(IN) :: commit
    REAL(db),INTENT(INOUT) :: r(:,:,:,:),t(:,:,:,:)
    REAL(db),INTENT(INOUT) :: c(:,:,:,:,:),s(:,:,:,:,:),j(:,:,:,:,:)
    CALL c_stage(handle,order,dt/hbc,MERGE(1,0,commit),C_LOC(density))
    r=r+density(:,:,:,1,:)
    t=t+density(:,:,:,2,:)
    c=c+density(:,:,:,3:5,:)
    s=s+density(:,:,:,6:8,:)
    j=j+density(:,:,:,9:11,:)
  END SUBROUTINE
  SUBROUTINE gpu_pull(ps)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(INOUT) :: ps(:,:,:,:,:)
    IF(gpu_enabled) CALL c_pull(handle,C_LOC(ps))
  END SUBROUTINE
  SUBROUTINE gpu_push(ps)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(IN) :: ps(:,:,:,:,:)
    IF(gpu_enabled) CALL c_push(handle,C_LOC(ps))
  END SUBROUTINE
  SUBROUTINE gpu_probe(h,den)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(OUT) :: h(:,:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(OUT) :: den(:,:,:,:,:)
    CALL c_probe(handle,C_LOC(h),C_LOC(den))
  END SUBROUTINE
  SUBROUTINE gpu_finish
    IF(gpu_enabled) CALL c_finish(handle)
    handle=C_NULL_PTR
    gpu_enabled=.FALSE.
    IF(ALLOCATED(density)) DEALLOCATE(density)
  END SUBROUTINE
END MODULE
