CXX=mpic++
ifneq ($(wildcard ${SMARTIES_ROOT}/extern/.*),)
CXX=${SMARTIES_ROOT}/extern/bin/mpic++
endif

LIBS= -L${SMARTIES_ROOT}/lib -lsmarties
CPPFLAGS= -std=c++14 -Wall -O2 -I${SMARTIES_ROOT}/include
CPPFLAGS+= -Wall -Wextra 

.DEFAULT_GOAL := exec

exec: main.o
	$(CXX) $(extra) $< ${LIBS} -o $@

main.o: main.cpp
	$(CXX) $(extra) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f *.o *.s exec
	rm -rf *.raw *.log *.txt *.dat smarties_AFUNIX_* simulation_*
