#include <stdint.h>

#define csr_tohost(csr_val) { \
    asm volatile ("csrw 0x51e,%[v]" :: [v]"r"(csr_val)); \
}

#define N 2048

static inline uint32_t mix(uint32_t x) {
    x ^= (x << 13);
    x ^= (x >> 17);
    x ^= (x << 5);
    return x;
}

void main() {
    uint32_t acc0 = 0x12345678u;
    uint32_t acc1 = 0x9e3779b9u;
    uint32_t acc2 = 0xa5a5a5a5u;
    uint32_t acc3 = 0x0f1e2d3cu;

    for (uint32_t i = 1; i <= N; i++) {
        // Four mostly independent update chains to expose ILP.
        acc0 = (acc0 + (i * 3u)) ^ (acc1 << 1);
        acc1 = (acc1 + (i * 5u)) ^ (acc2 >> 1);
        acc2 = (acc2 + (i * 7u)) ^ (acc3 << 2);
        acc3 = (acc3 + (i * 11u)) ^ (acc0 >> 2);

        acc0 = mix(acc0);
        acc1 = mix(acc1);
        acc2 = mix(acc2);
        acc3 = mix(acc3);
    }

    uint32_t out = acc0 ^ acc1 ^ acc2 ^ acc3;
    if (out == 0x3f67300cu) {
        csr_tohost(1);
    } else {
        csr_tohost(2);
    }

    for (;;) {
        asm volatile ("nop");
    }
}
