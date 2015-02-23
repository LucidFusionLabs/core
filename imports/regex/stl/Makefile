CPLUS=g++
CFLAGS=-I.. -DREGEXP_UNICODE
#CFLAGS=-I.. 
LD=g++
CC=gcc

CPL=$(CPLUS) $(CFLAGS)
CCL=$(CC) $(CFLAGS)

default: retest

clean:
	rm *.o retest

stdregexp.o: stdregexp.cpp stdregexp.hpp ../regexp.h
	$(CPL) -c stdregexp.cpp

regexp.o: ../regexp.c
	$(CCL) -c ../regexp.c

memory.o: ../memory.c
	$(CCL) -c ../memory.c

regerror.o: ../regerror.c
	$(CCL) -c ../regerror.c

report.o: ../report.c
	$(CCL) -c ../report.c
	
widechar.o: ../widechar.c
	$(CCL) -c ../widechar.c

retest.o: retest.cpp stdregexp.hpp ../regexp.h
	$(CPL) -c retest.cpp

retest: stdregexp.o regexp.o memory.o regerror.o report.o widechar.o retest.o
	$(LD) -o retest stdregexp.o regexp.o memory.o regerror.o retest.o report.o widechar.o


