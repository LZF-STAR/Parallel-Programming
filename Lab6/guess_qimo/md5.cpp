#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <immintrin.h>  // SSE/AVX指令集

using namespace std;
using namespace chrono;

// SIMD版本的MD5函数
#define SIMD_F(x, y, z) (_mm_or_si128(_mm_and_si128(x, y), _mm_andnot_si128(x, z)))
#define SIMD_G(x, y, z) (_mm_or_si128(_mm_and_si128(x, z), _mm_andnot_si128(z, y)))
#define SIMD_H(x, y, z) (_mm_xor_si128(_mm_xor_si128(x, y), z))
#define SIMD_I(x, y, z) (_mm_xor_si128(y, _mm_or_si128(x, _mm_xor_si128(z, _mm_set1_epi32(0xFFFFFFFF)))))

// SIMD循环左移
inline __m128i SIMD_ROTATELEFT(__m128i x, int n) {
    return _mm_or_si128(_mm_slli_epi32(x, n), _mm_srli_epi32(x, 32 - n));
}

// 原始的单消息处理函数（保留兼容性）
Byte *StringProcess(string input, int *n_byte)
{
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();
	int bitLength = length * 8;

	int paddingBits = bitLength % 512;
	if (paddingBits > 448) {
		paddingBits = 512 - (paddingBits - 448);
	} else if (paddingBits < 448) {
		paddingBits = 448 - paddingBits;
	} else if (paddingBits == 448) {
		paddingBits = 512;
	}

	int paddingBytes = paddingBits / 8;
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	memcpy(paddedMessage, blocks, length);
	paddedMessage[length] = 0x80;
	memset(paddedMessage + length + 1, 0, paddingBytes - 1);

	for (int i = 0; i < 8; ++i) {
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	int residual = 8 * paddedLength % 512;
	assert(residual == 0);

	*n_byte = paddedLength;
	return paddedMessage;
}

// SIMD批量字符串预处理
void BatchStringProcess(const char** inputs, int count, Byte** paddedMessages, int* messageLengths) {
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        string input(inputs[i]);
        paddedMessages[i] = StringProcess(input, &messageLengths[i]);
    }
}

// SIMD版本的MD5轮函数
#define SIMD_FF(a, b, c, d, x, s, ac) { \
    a = _mm_add_epi32(a, x); \
    a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, SIMD_F(b, c, d)); \
    a = SIMD_ROTATELEFT(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define SIMD_GG(a, b, c, d, x, s, ac) { \
    a = _mm_add_epi32(a, x); \
    a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, SIMD_G(b, c, d)); \
    a = SIMD_ROTATELEFT(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define SIMD_HH(a, b, c, d, x, s, ac) { \
    a = _mm_add_epi32(a, x); \
    a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, SIMD_H(b, c, d)); \
    a = SIMD_ROTATELEFT(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define SIMD_II(a, b, c, d, x, s, ac) { \
    a = _mm_add_epi32(a, x); \
    a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, SIMD_I(b, c, d)); \
    a = SIMD_ROTATELEFT(a, s); \
    a = _mm_add_epi32(a, b); \
}

// SIMD批量MD5计算 - 同时处理4个消息
void SIMD_MD5_Batch4(Byte** paddedMessages, int* messageLengths, bit32 results[4][4]) {
    // 初始化4个MD5状态
    __m128i va = _mm_set1_epi32(0x67452301);
    __m128i vb = _mm_set1_epi32(0xefcdab89);
    __m128i vc = _mm_set1_epi32(0x98badcfe);
    __m128i vd = _mm_set1_epi32(0x10325476);
    
    // 假设所有消息长度相同（简化处理）
    int n_blocks = messageLengths[0] / 64;
    
    for (int block = 0; block < n_blocks; block++) {
        __m128i x[16];
        
        // 数据重组：从4个消息中提取对应的块
        for (int i = 0; i < 16; i++) {
            uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
            
            for (int j = 0; j < 4; j++) {
                uint32_t value = (paddedMessages[0][block * 64 + i * 4 + j] << (j * 8));
                w0 |= value;
                value = (paddedMessages[1][block * 64 + i * 4 + j] << (j * 8));
                w1 |= value;
                value = (paddedMessages[2][block * 64 + i * 4 + j] << (j * 8));
                w2 |= value;
                value = (paddedMessages[3][block * 64 + i * 4 + j] << (j * 8));
                w3 |= value;
            }
            
            x[i] = _mm_set_epi32(w3, w2, w1, w0);
        }
        
        __m128i aa = va, bb = vb, cc = vc, dd = vd;
        
        // Round 1 - SIMD版本
        SIMD_FF(aa, bb, cc, dd, x[0], s11, 0xd76aa478);
        SIMD_FF(dd, aa, bb, cc, x[1], s12, 0xe8c7b756);
        SIMD_FF(cc, dd, aa, bb, x[2], s13, 0x242070db);
        SIMD_FF(bb, cc, dd, aa, x[3], s14, 0xc1bdceee);
        SIMD_FF(aa, bb, cc, dd, x[4], s11, 0xf57c0faf);
        SIMD_FF(dd, aa, bb, cc, x[5], s12, 0x4787c62a);
        SIMD_FF(cc, dd, aa, bb, x[6], s13, 0xa8304613);
        SIMD_FF(bb, cc, dd, aa, x[7], s14, 0xfd469501);
        SIMD_FF(aa, bb, cc, dd, x[8], s11, 0x698098d8);
        SIMD_FF(dd, aa, bb, cc, x[9], s12, 0x8b44f7af);
        SIMD_FF(cc, dd, aa, bb, x[10], s13, 0xffff5bb1);
        SIMD_FF(bb, cc, dd, aa, x[11], s14, 0x895cd7be);
        SIMD_FF(aa, bb, cc, dd, x[12], s11, 0x6b901122);
        SIMD_FF(dd, aa, bb, cc, x[13], s12, 0xfd987193);
        SIMD_FF(cc, dd, aa, bb, x[14], s13, 0xa679438e);
        SIMD_FF(bb, cc, dd, aa, x[15], s14, 0x49b40821);
        
        // Round 2 - SIMD版本
        SIMD_GG(aa, bb, cc, dd, x[1], s21, 0xf61e2562);
        SIMD_GG(dd, aa, bb, cc, x[6], s22, 0xc040b340);
        SIMD_GG(cc, dd, aa, bb, x[11], s23, 0x265e5a51);
        SIMD_GG(bb, cc, dd, aa, x[0], s24, 0xe9b6c7aa);
        SIMD_GG(aa, bb, cc, dd, x[5], s21, 0xd62f105d);
        SIMD_GG(dd, aa, bb, cc, x[10], s22, 0x2441453);
        SIMD_GG(cc, dd, aa, bb, x[15], s23, 0xd8a1e681);
        SIMD_GG(bb, cc, dd, aa, x[4], s24, 0xe7d3fbc8);
        SIMD_GG(aa, bb, cc, dd, x[9], s21, 0x21e1cde6);
        SIMD_GG(dd, aa, bb, cc, x[14], s22, 0xc33707d6);
        SIMD_GG(cc, dd, aa, bb, x[3], s23, 0xf4d50d87);
        SIMD_GG(bb, cc, dd, aa, x[8], s24, 0x455a14ed);
        SIMD_GG(aa, bb, cc, dd, x[13], s21, 0xa9e3e905);
        SIMD_GG(dd, aa, bb, cc, x[2], s22, 0xfcefa3f8);
        SIMD_GG(cc, dd, aa, bb, x[7], s23, 0x676f02d9);
        SIMD_GG(bb, cc, dd, aa, x[12], s24, 0x8d2a4c8a);
        
        // Round 3 - SIMD版本
        SIMD_HH(aa, bb, cc, dd, x[5], s31, 0xfffa3942);
        SIMD_HH(dd, aa, bb, cc, x[8], s32, 0x8771f681);
        SIMD_HH(cc, dd, aa, bb, x[11], s33, 0x6d9d6122);
        SIMD_HH(bb, cc, dd, aa, x[14], s34, 0xfde5380c);
        SIMD_HH(aa, bb, cc, dd, x[1], s31, 0xa4beea44);
        SIMD_HH(dd, aa, bb, cc, x[4], s32, 0x4bdecfa9);
        SIMD_HH(cc, dd, aa, bb, x[7], s33, 0xf6bb4b60);
        SIMD_HH(bb, cc, dd, aa, x[10], s34, 0xbebfbc70);
        SIMD_HH(aa, bb, cc, dd, x[13], s31, 0x289b7ec6);
        SIMD_HH(dd, aa, bb, cc, x[0], s32, 0xeaa127fa);
        SIMD_HH(cc, dd, aa, bb, x[3], s33, 0xd4ef3085);
        SIMD_HH(bb, cc, dd, aa, x[6], s34, 0x4881d05);
        SIMD_HH(aa, bb, cc, dd, x[9], s31, 0xd9d4d039);
        SIMD_HH(dd, aa, bb, cc, x[12], s32, 0xe6db99e5);
        SIMD_HH(cc, dd, aa, bb, x[15], s33, 0x1fa27cf8);
        SIMD_HH(bb, cc, dd, aa, x[2], s34, 0xc4ac5665);
        
        // Round 4 - SIMD版本
        SIMD_II(aa, bb, cc, dd, x[0], s41, 0xf4292244);
        SIMD_II(dd, aa, bb, cc, x[7], s42, 0x432aff97);
        SIMD_II(cc, dd, aa, bb, x[14], s43, 0xab9423a7);
        SIMD_II(bb, cc, dd, aa, x[5], s44, 0xfc93a039);
        SIMD_II(aa, bb, cc, dd, x[12], s41, 0x655b59c3);
        SIMD_II(dd, aa, bb, cc, x[3], s42, 0x8f0ccc92);
        SIMD_II(cc, dd, aa, bb, x[10], s43, 0xffeff47d);
        SIMD_II(bb, cc, dd, aa, x[1], s44, 0x85845dd1);
        SIMD_II(aa, bb, cc, dd, x[8], s41, 0x6fa87e4f);
        SIMD_II(dd, aa, bb, cc, x[15], s42, 0xfe2ce6e0);
        SIMD_II(cc, dd, aa, bb, x[6], s43, 0xa3014314);
        SIMD_II(bb, cc, dd, aa, x[13], s44, 0x4e0811a1);
        SIMD_II(aa, bb, cc, dd, x[4], s41, 0xf7537e82);
        SIMD_II(dd, aa, bb, cc, x[11], s42, 0xbd3af235);
        SIMD_II(cc, dd, aa, bb, x[2], s43, 0x2ad7d2bb);
        SIMD_II(bb, cc, dd, aa, x[9], s44, 0xeb86d391);
        
        va = _mm_add_epi32(va, aa);
        vb = _mm_add_epi32(vb, bb);
        vc = _mm_add_epi32(vc, cc);
        vd = _mm_add_epi32(vd, dd);
    }
    
    // 提取结果
    alignas(16) uint32_t temp[4];
    _mm_store_si128((__m128i*)temp, va);
    for (int i = 0; i < 4; i++) results[i][0] = temp[i];
    _mm_store_si128((__m128i*)temp, vb);
    for (int i = 0; i < 4; i++) results[i][1] = temp[i];
    _mm_store_si128((__m128i*)temp, vc);
    for (int i = 0; i < 4; i++) results[i][2] = temp[i];
    _mm_store_si128((__m128i*)temp, vd);
    for (int i = 0; i < 4; i++) results[i][3] = temp[i];
    
    // 字节顺序反转
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t value = results[i][j];
            results[i][j] = ((value & 0xff) << 24) |
                           ((value & 0xff00) << 8) |
                           ((value & 0xff0000) >> 8) |
                           ((value & 0xff000000) >> 24);
        }
    }
}

// 保留原始接口的兼容性
void MD5Hash(string input, bit32 *state) {
    Byte *paddedMessage;
    int messageLength;
    
    paddedMessage = StringProcess(input, &messageLength);
    int n_blocks = messageLength / 64;
    
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;
    
    for (int i = 0; i < n_blocks; i += 1) {
        bit32 x[16];
        
        for (int i1 = 0; i1 < 16; ++i1) {
            x[i1] = (paddedMessage[4 * i1 + i * 64]) |
                    (paddedMessage[4 * i1 + 1 + i * 64] << 8) |
                    (paddedMessage[4 * i1 + 2 + i * 64] << 16) |
                    (paddedMessage[4 * i1 + 3 + i * 64] << 24);
        }
        
        bit32 a = state[0], b = state[1], c = state[2], d = state[3];
        
        // Round 1
        FF(a, b, c, d, x[0], s11, 0xd76aa478);
        FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);
        FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);
        FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);
        FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122);
        FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e);
        FF(b, c, d, a, x[15], s14, 0x49b40821);
        
        // Round 2
        GG(a, b, c, d, x[1], s21, 0xf61e2562);
        GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51);
        GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);
        GG(d, a, b, c, x[10], s22, 0x2441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);
        GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);
        
        // Round 3
        HH(a, b, c, d, x[5], s31, 0xfffa3942);
        HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);
        HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH(b, c, d, a, x[6], s34, 0x4881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH(b, c, d, a, x[2], s34, 0xc4ac5665);
        
        // Round 4
        II(a, b, c, d, x[0], s41, 0xf4292244);
        II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7);
        II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3);
        II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d);
        II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);
        II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);
        II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II(b, c, d, a, x[9], s44, 0xeb86d391);
        
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }
    
    for (int i = 0; i < 4; i++) {
        uint32_t value = state[i];
        state[i] = ((value & 0xff) << 24) |
                   ((value & 0xff00) << 8) |
                   ((value & 0xff0000) >> 8) |
                   ((value & 0xff000000) >> 24);
    }
    
    delete[] paddedMessage;
}

// 批量MD5计算接口 - 自动处理非4倍数
void BatchMD5Hash(const vector<string>& inputs, vector<bit32*>& outputs) {
    size_t count = inputs.size();
    size_t full_batches = count / 4;
    size_t remainder = count % 4;
    
    // 处理完整的4个一组
    #pragma omp parallel for
    for (size_t batch = 0; batch < full_batches; batch++) {
        const char* batch_inputs[4];
        Byte* padded[4];
        int lengths[4];
        bit32 results[4][4];
        
        // 准备批次数据
        for (int i = 0; i < 4; i++) {
            batch_inputs[i] = inputs[batch * 4 + i].c_str();
        }
        
        // 批量预处理
        BatchStringProcess(batch_inputs, 4, padded, lengths);
        
        // SIMD批量计算
        SIMD_MD5_Batch4(padded, lengths, results);
        
        // 存储结果
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                outputs[batch * 4 + i][j] = results[i][j];
            }
            delete[] padded[i];
        }
    }
    
    // 处理剩余的（串行）
    for (size_t i = full_batches * 4; i < count; i++) {
        MD5Hash(inputs[i], outputs[i]);
    }
}