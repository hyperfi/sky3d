# Project-local GNU/OpenMP build of the existing Sky3D source tree.
# Run from WSL through ../scripts/build_wsl.sh.

FC := gfortran
SRC := ../../../Code
EXEC := sky3d.omp
FFLAGS := -O3 -fopenmp -msse4.2 -mfpmath=sse -ffast-math \
	-finline-functions -funroll-loops
LDLIBS := -lfftw3 -llapack -lopenblas

OBJS := params.o gpu_runtime.o grids.o levels.o fourier.o forces.o sequential.o \
	inout.o coulomb.o trivial.o densities.o fragments.o twobody.o \
	energies.o static.o meanfield.o dynamic.o pairs.o moment.o \
	main3d.o user.o external.o ylm.o

.PHONY: all clean provenance

all: $(EXEC)

$(EXEC): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $(OBJS) $(LDLIBS)

user.o: $(SRC)/user.f90 params.o grids.o levels.o
ylm.o: $(SRC)/ylm.f90 params.o
external.o: $(SRC)/external.f90 params.o sequential.o grids.o levels.o meanfield.o ylm.o moment.o
coulomb.o: $(SRC)/coulomb.f90 params.o grids.o densities.o
inout.o: $(SRC)/inout.f90 params.o sequential.o grids.o moment.o densities.o levels.o coulomb.o forces.o meanfield.o
trivial.o: $(SRC)/trivial.f90 params.o grids.o
forces.o: $(SRC)/forces.f90 $(SRC)/forces.data params.o
fourier.o: $(SRC)/fourier.f90 params.o grids.o
params.o: $(SRC)/params.f90
gpu_runtime.o: $(SRC)/gpu_runtime.f90 params.o
moment.o: $(SRC)/moment.f90 params.o grids.o densities.o ylm.o
sequential.o: $(SRC)/sequential.f90 levels.o params.o grids.o densities.o
levels.o: $(SRC)/levels.f90 params.o grids.o forces.o fourier.o trivial.o
grids.o: $(SRC)/grids.f90 params.o forces.o
densities.o: $(SRC)/densities.f90 params.o grids.o levels.o trivial.o
fragments.o: $(SRC)/fragments.f90 params.o grids.o forces.o levels.o sequential.o
twobody.o: $(SRC)/twobody.f90 params.o grids.o densities.o moment.o forces.o
energies.o: $(SRC)/energies.f90 params.o forces.o densities.o levels.o grids.o trivial.o pairs.o
meanfield.o: $(SRC)/meanfield.f90 params.o densities.o forces.o grids.o coulomb.o trivial.o levels.o gpu_runtime.o
pairs.o: $(SRC)/pairs.f90 forces.o grids.o levels.o densities.o
static.o: $(SRC)/static.f90 params.o densities.o grids.o levels.o moment.o energies.o inout.o pairs.o meanfield.o
dynamic.o: $(SRC)/dynamic.f90 params.o densities.o grids.o trivial.o levels.o moment.o energies.o inout.o meanfield.o sequential.o twobody.o external.o
main3d.o: $(SRC)/main3d.f90 params.o fourier.o forces.o densities.o meanfield.o levels.o grids.o fragments.o sequential.o dynamic.o static.o coulomb.o user.o

%.o:
	$(FC) $(FFLAGS) -I$(SRC) -I. -J. -c $< -o $@

provenance: $(EXEC)
	@$(FC) --version | head -n 1
	@sha256sum $(EXEC)
	@file $(EXEC)
	@ldd $(EXEC)

clean:
	rm -f -- *.o *.mod $(EXEC)
