#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <arm_neon.h>
#include<string>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main

// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    uint32x4_t state[4];
    string pw = "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    MD5Hash(pw,pw,pw,pw,state);
    uint32_t outA[4],outB[4], outC[4], outD[4];
    vst1q_u32(outA,state[0]);
    vst1q_u32(outB,state[1]);
    vst1q_u32(outC,state[2]);
    vst1q_u32(outD,state[3]);
    for (int i1=0;i1<4;i1+=1)
    {
        bit32 a=outA[i1];
        bit32 b=outB[i1];
        bit32 c=outC[i1];
        bit32 d=outD[i1];
        cout << setw(8) << setfill('0') << hex << a
             << setw(8) << setfill('0') << hex << b
             << setw(8) << setfill('0') << hex << c
             << setw(8) << setfill('0') << hex << d << endl;
    }
    cout << endl;
}