CC = riscv64-unknown-elf-gcc
CFLAGS = -O3 -Wall -funroll-loops -ffast-math
TARGET = runtime/inference

all: $(TARGET)

$(TARGET): inference.c runtime/weights.h runtime/test_images.h
	$(CC) $(CFLAGS) -o $(TARGET) inference.c -Iruntime

run: $(TARGET)
	spike pk $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean