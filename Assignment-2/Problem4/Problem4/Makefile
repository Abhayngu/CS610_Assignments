
program : test.o pkeystore.o
	cc -g -o program test.o pkeystore.o -lpthread

test.o : test.c pkeystore.h
	cc -g -c test.c -lpthread

pkeystore.o : pkeystore.c pkeystore.h
	cc -g -c pkeystore.c -lpthread

clean :
	rm program test.o pkeystore.o
