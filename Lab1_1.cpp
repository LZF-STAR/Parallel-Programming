#include<iostream>
#include<windows.h>
#include<stdlib.h>
#include<vector>
using namespace std;
const int N = 15000;

//平凡算法,逐列访问
double ord(const double* a, const double* b, double* ans,int n,int r) {
	double total_time;
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int k = 0; k < r; k++) {
		for (int i = 0; i < n; i++) {
			ans[i] = 0.0;
			for (int j = 0; j < n; j++) {
				ans[i] += a[j * n + i] * b[i]; //通过线性化的索引访问二维数组a中的元素
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	total_time = (tail - head) * 1000.0 / freq;//单位ms
	double res = total_time / r;
	return res;
}

//cache优化算法，c++为行主存模式
//按行访问元素的顺序通常更符合内存中的存储布局，从而提高cache的命中率
//类比ppt中的代码
double use_cache(const double* a, const double* b, double* ans,int n,int r) {
	double total_time;
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int k = 0; k < r; k++) {
		for (int i = 0; i < n; i++) {
			ans[i] = 0.0;
		}
		for (int j = 0; j < n; j++) {
			const double* row = a + j * n; //定位到第j行的起始地址
			for (int i = 0; i < n; i++) {
				ans[i] += row[i] * b[j];
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	total_time = (tail - head) * 1000.0 / freq;//单位ms
	double res = total_time / r;
	return res;
}
//初始化
void init(double* a,int n) {
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}
}
int main() {
	double* a = new double[N * N]; //n*n矩阵
	double* b = new double[N]; //向量
	double* ans1 = new double[N]; //平凡算法结果
	double* ans2 = new double[N]; //优化算法结果
	//初始化
	init(a, N * N);
	init(b, N);
	vector<double> ress1, ress2;
	for (int i = 1000; i <= 15000; i += 1000) {
		ress1.push_back(ord(a,b,ans1,i,5));//5次实验
		ress2.push_back(use_cache(a,b,ans2,i,5));
	}
	cout << "接下来依次输出的是：平凡算法所用时间（ms），优化算法所用时间（ms），运行速度提升倍数";
	cout << endl;
	cout << "从上至下依次是1000，2000，...，15000规模的数据";
	cout << endl;
	for (int i = 0; i < ress1.size(); i++) {
		double bei = ress1[i] / ress2[i];
		cout << ress1[i] << "        " << ress2[i] <<"              " << bei;
		cout << endl;
	}
	delete[] a;
	delete[] b;
	delete[] ans1;
	delete[] ans2;
}