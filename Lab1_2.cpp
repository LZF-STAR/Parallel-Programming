#include<iostream>
#include<windows.h>
#include<stdlib.h>
#include<vector>
using namespace std;
const int N = 1e5;

//平凡算法,逐个元素访问
double ord(int *a,int n,int r) {
    double total_time;
	long long sum=0;
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int k = 0; k < r; k++) {
		for (int i = 0; i < n; i++) {
			sum+=a[i];
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	total_time = (tail - head) * 1000.0 / freq;//单位ms
	return total_time;
}

//优化算法(四路链式累加)
double opt(int *a,int n,int r) {
	double total_time;
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int k = 0; k < r; k++) {
		long long sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        int i;
		//四路链式累加
		for (i = 0; i <= n - 4; i += 4) {
			sum1 += a[i];
			sum2 += a[i+1];
			sum3 += a[i+2];
			sum4 += a[i+3];
		}
        long long sum = sum1 + sum2 + sum3 + sum4;
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	total_time = (tail - head) * 1000.0 / freq;//单位ms
	return total_time;
}


//初始化
void init(int* a,int n) {
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}
}
int main() {
	int* a = new int[N];
	int* b = new int[N];
	//初始化
	init(a, N);
	init(b, N);
	vector<double> ress1, ress2;
	for (int i = 1e4; i <= 1e5; i += 1e4) {
		ress1.push_back(ord(a,i,1e5));//1e5次实验
		ress2.push_back(opt(b,i,1e5));
	}
	cout << "接下来依次输出的是：平凡算法所用时间（ms），优化算法所用时间（ms），运行速度提升倍数";
	cout << endl;
	cout << "从上至下依次是1e4，2e4，...，1e5规模的数据";
	cout << endl;
	for (int i = 0; i < ress1.size(); i++) {
		double bei = ress1[i] / ress2[i];
		cout << ress1[i] << "        " << ress2[i] <<"              " << bei;
		cout << endl;
	}
	delete[] a;
	delete[] b;
}