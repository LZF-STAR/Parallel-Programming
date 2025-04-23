#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <emmintrin.h>
#include <tmmintrin.h>
#include<string>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -mssse3 -o main

// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    __m128i state[4];
    string batch[4]; 
    for(int i=0;i<4;i++){
        batch[i] = "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
        ;
    }
    MD5Hash(batch,state);
    alignas(16) uint32_t outA[4], outB[4], outC[4], outD[4];
    _mm_storeu_si128((__m128i *)outA, state[0]);
    _mm_storeu_si128((__m128i *)outB, state[1]);
    _mm_storeu_si128((__m128i *)outC, state[2]);
    _mm_storeu_si128((__m128i *)outD, state[3]);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        bit32 a = outA[i1];
        bit32 b = outB[i1];
        bit32 c = outC[i1];
        bit32 d = outD[i1];
        cout << setw(8) << setfill('0') << hex << a
             << setw(8) << setfill('0') << hex << b
             << setw(8) << setfill('0') << hex << c
             << setw(8) << setfill('0') << hex << d << endl;
    }
    cout << endl;
}