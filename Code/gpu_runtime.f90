! CPU-only implementation of the optional GPU runtime interface.
MODULE GPU_Runtime
  USE Params, ONLY: db
  IMPLICIT NONE
  LOGICAL :: gpu_resident=.FALSE.,gpu_diagnostics_enabled=.FALSE.
  LOGICAL :: gpu_enabled=.FALSE.
  LOGICAL :: gpu_fields_enabled=.FALSE.
CONTAINS
  SUBROUTINE gpu_coulomb_config(q,periodic,scale)
    COMPLEX(db),INTENT(IN) :: q(:,:,:)
    LOGICAL,INTENT(IN) :: periodic
    REAL(db),INTENT(IN) :: scale
  END SUBROUTINE
  SUBROUTINE gpu_coulomb_pull(wc)
    REAL(db),INTENT(OUT) :: wc(:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_skyrme(f,r,t,c,sd,j,u,b,s,dbm,a,w)
    REAL(db),INTENT(IN) :: f(:),r(:,:,:,:),t(:,:,:,:)
    REAL(db),INTENT(IN) :: c(:,:,:,:,:),sd(:,:,:,:,:),j(:,:,:,:,:)
    REAL(db),INTENT(OUT) :: u(:,:,:,:),b(:,:,:,:)
    REAL(db),INTENT(OUT) :: s(:,:,:,:,:),dbm(:,:,:,:,:),a(:,:,:,:,:),w(:,:,:,:,:)
    ERROR STOP 'GPU fields called in CPU build'
  END SUBROUTINE
  SUBROUTINE gpu_initialize(ps,weights,iq,spacing,tfft,tmpi)
    COMPLEX(db),INTENT(INOUT) :: ps(:,:,:,:,:)
    REAL(db),INTENT(IN) :: weights(:),spacing(3)
    INTEGER,INTENT(IN) :: iq(:)
    LOGICAL,INTENT(IN) :: tfft,tmpi
    CHARACTER(32) :: backend
    CALL get_environment_variable('SKY3D_BACKEND',backend)
    IF(TRIM(backend)=='gpu') ERROR STOP 'This executable has no GPU backend; build CUDA/ first'
    IF(LEN_TRIM(backend)>0.AND.TRIM(backend)/='cpu') ERROR STOP 'SKY3D_BACKEND must be cpu or gpu'
  END SUBROUTINE
  SUBROUTINE gpu_fields(u,b,s,dbm,a,w)
    REAL(db),INTENT(IN) :: u(:,:,:,:),b(:,:,:,:)
    REAL(db),INTENT(IN) :: s(:,:,:,:,:),dbm(:,:,:,:,:),a(:,:,:,:,:),w(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_stage(order,dt,hbc,commit,r,t,c,s,j)
    INTEGER,INTENT(IN) :: order
    REAL(db),INTENT(IN) :: dt,hbc
    LOGICAL,INTENT(IN) :: commit
    REAL(db),INTENT(INOUT) :: r(:,:,:,:),t(:,:,:,:)
    REAL(db),INTENT(INOUT) :: c(:,:,:,:,:),s(:,:,:,:,:),j(:,:,:,:,:)
    ERROR STOP 'GPU stage called in CPU build'
  END SUBROUTINE
  SUBROUTINE gpu_pull(ps)
    COMPLEX(db),INTENT(INOUT) :: ps(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_push(ps)
    COMPLEX(db),INTENT(IN) :: ps(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_probe(h,den)
    COMPLEX(db),INTENT(OUT) :: h(:,:,:,:,:)
    REAL(db),INTENT(OUT) :: den(:,:,:,:,:)
    ERROR STOP 'GPU probe called in CPU build'
  END SUBROUTINE
  SUBROUTINE gpu_fields_pull(u,b,s,dbm,a,w)
    REAL(db),CONTIGUOUS,TARGET,INTENT(OUT) :: u(:,:,:,:),b(:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(OUT) :: s(:,:,:,:,:),dbm(:,:,:,:,:),a(:,:,:,:,:),w(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE assign_density(r,t,c,s,j)
    REAL(db),INTENT(OUT) :: r(:,:,:,:),t(:,:,:,:),c(:,:,:,:,:),s(:,:,:,:,:),j(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_download_density(r,t,c,s,j)
    REAL(db),INTENT(OUT) :: r(:,:,:,:),t(:,:,:,:),c(:,:,:,:,:),s(:,:,:,:,:),j(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_rebuild_density(ps,weights,r,t,c,s,j)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(IN) :: ps(:,:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(IN) :: weights(:)
    REAL(db),INTENT(OUT) :: r(:,:,:,:),t(:,:,:,:),c(:,:,:,:,:),s(:,:,:,:,:),j(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_properties(cm,h2m,orbital,spin,kin,parity,energy,norm)
    REAL(db),TARGET,INTENT(IN) :: cm(3),h2m(2)
    REAL(db),INTENT(OUT) :: orbital(:,:),spin(:,:),kin(:),parity(:),energy(:),norm(:)
  END SUBROUTINE
  SUBROUTINE gpu_moments_first(values)
    REAL(db),TARGET,INTENT(OUT) :: values(7,2)
  END SUBROUTINE
  SUBROUTINE gpu_moments_second(cm,values)
    REAL(db),TARGET,INTENT(IN) :: cm(3,2)
    REAL(db),TARGET,INTENT(OUT) :: values(19,2)
  END SUBROUTINE
  SUBROUTINE gpu_static_step(ps,spe,e0,h2m,x0,f1,f2,norm,delta)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(INOUT) :: ps(:,:,:,:,:)
    REAL(db),CONTIGUOUS,TARGET,INTENT(INOUT) :: spe(:)
    REAL(db),INTENT(IN) :: e0,h2m,x0
    REAL(db),CONTIGUOUS,TARGET,INTENT(OUT) :: f1(:),f2(:),norm(:),delta(:)
  END SUBROUTINE
  SUBROUTINE gpu_hamiltonian(ps,h)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(IN) :: ps(:,:,:,:,:)
    COMPLEX(db),CONTIGUOUS,TARGET,INTENT(OUT) :: h(:,:,:,:,:)
  END SUBROUTINE
  SUBROUTINE gpu_finish
  END SUBROUTINE
END MODULE
