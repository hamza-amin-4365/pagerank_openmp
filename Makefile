CC = gcc
CFLAGS = -O3 -fopenmp
LIBS = -lm
TARGET = pagerank
SRC = pagerank.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

run-1:
	OMP_NUM_THREADS=1 ./$(TARGET) data/web-Stanford.txt

run-2:
	OMP_NUM_THREADS=2 ./$(TARGET) data/web-Stanford.txt

run-4:
	OMP_NUM_THREADS=4 ./$(TARGET) data/web-Stanford.txt

run-8:
	OMP_NUM_THREADS=8 ./$(TARGET) data/web-Stanford.txt

run-16:
	OMP_NUM_THREADS=16 ./$(TARGET) data/web-Stanford.txt

clean:
	rm -f $(TARGET) *.o

test:
	echo "Not Implemented yet"

.PHONY: all run-1 run-2 run-4 run-8 run-16 clean test
