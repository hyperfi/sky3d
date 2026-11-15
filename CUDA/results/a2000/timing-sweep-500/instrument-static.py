from pathlib import Path
for backend in ("cpu","gpu"):
    path=Path(backend)/"static.f90"
    src=path.read_text()
    start="  SUBROUTINE statichf\n"
    assert src.count(start)==1
    src=src.replace(start,start+"    USE omp_lib, ONLY: omp_get_wtime\n    REAL(db) :: static_loop_start\n")
    start="    Iteration: DO iter=firstiter,maxiter"
    assert src.count(start)==1
    src=src.replace(start,"    static_loop_start=omp_get_wtime()\n"+start)
    end="    END DO Iteration\n"
    assert src.count(end)==1
    src=src.replace(end,end+'''    IF(wflag) WRITE(*,'(A,ES24.16,A,I0)') 'Static loop wall seconds: ', &
         omp_get_wtime()-static_loop_start,'; iterations: ',MIN(iter,maxiter)-firstiter+1
''')
    path.write_text(src)
