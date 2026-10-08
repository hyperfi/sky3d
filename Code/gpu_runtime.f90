! CPU-only implementation of the optional GPU runtime interface.
MODULE GPU_Runtime
  USE Params, ONLY: db
  IMPLICIT NONE
  LOGICAL :: gpu_enabled=.FALSE.
CONTAINS
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
  SUBROUTINE gpu_finish
  END SUBROUTINE
END MODULE
