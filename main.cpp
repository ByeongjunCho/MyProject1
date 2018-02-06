#include "Vector.h"

void test1()
{
	double b[4] = { 1, 2, 5, 7 };
	double c[4] = { 1, 4, 8, 2 };
	double result;
	double a[] = { 0,0,0,1,1,0,1,1 };

	int i;
	i = 0;

	int n_all = 1;
	int n_model_inputs = 2;
	double outputs[4];
	int ntarg = 4;
	int *nhid_all = {};
	double *weigths_op;

	double *hid_act;
	double final_layer_weigths[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	int classifier = 0;


	for (i; i < 8; i++) {
		printf("%4.5f ", a[i]);
	}
	result = dotprod(4, c, b);
	printf("\n This is the result : %4.5f \n ", result);

	trial_thr(a, n_all, n_model_inputs, outputs, ntarg, nhid_all, &weigths_op, &hid_act,
		final_layer_weigths, classifier);

	for (int i = 0; i < 4; i++)
		cout << outputs[i] << "  ";
}
void test2()
{
	double a[] = { 0.5,1 };

	int i;
	i = 0;

	double w00[3] = { 0.2,2,0 };
	double w01[3] = { -0.1,1,0 };
	int n_all = 2;
	int n_model_inputs = 2;
	double outputs[2];
	int ntarg = 2;
	int nhid_all[] = { 2 };
	double w1[6] = { 0.2,2,0,-0.1,1,0 };
	double *weigths_op[] = { w1 };

	double h1[2];
	double *hid_act[] = { h1 };
	double final_layer_weigths[] = { 0.1,-0.1,0,0.3,0.4,0 }; // final_layer_weights[] = {첫 번째 결과 뉴런에 대한 가중치 + bias, 두 번째.... +bias}
	int classifier = 0;
	/*입력 = {0.5,1}, 출력 = {0.50,0.64}
	가중치 w00,w01,w10,w11 = {0.2,-0.1,2,1}
	마지막 레이어 가중치 = {0.1,0.2,-0.1,0.4}
	각 뉴런 값 순서대로 히든레이어 = {0.9,0.72}*/

	for (i; i < 2; i++) {
		printf("%4.5f ", a[i]);
	}
	cout << endl;
	trial_thr(a, n_all, n_model_inputs, outputs, ntarg, nhid_all, weigths_op, hid_act,
		final_layer_weigths, classifier);

	for (int i = 0; i < 2; i++)
		cout << outputs[i] << "  ";
}
void test3()
{
	double a[] = { 0.5,1 };  // 입력 데이터
	double b[] = { 1,0 };   // 목표 행렬

	int istart = 0;
	int istop = 1;
	int n_all = 2;
	int n_all_weights = 8;
	int n_model_inputs = 2;
	double outputs[2];
	int ntarg = 2;
	int nhid_all[] = { 2 };
	double w1[6] = { 0.2,2,0,-0.1,1,0 };
	double *weigths_op[] = { w1 };

	double h1[2];
	double *hid_act[] = { h1 };
	int max_neurons = 2;
	double this_delta[2];  // 크기는 ntarg와 같음

	double final_layer_weigths[] = { 0.1,-0.1,0,0.3,0.4,0 }; // final_layer_weights[] = {첫 번째 결과 뉴런에 대한 가중치 + bias, 두 번째.... +bias}
	int classifier = 0;
	/*입력 = {0.5,1}, 출력 = {0.50,0.64}
	가중치 w00,w01,w10,w11 = {0.2,-0.1,2,1}
	마지막 레이어 가중치 = {0.1,0.2,-0.1,0.4}
	각 뉴런 값 순서대로 히든레이어 = {0.9,0.72}*/

	for (int i; i < 2; i++) {
		printf("%4.5f ", a[i]);
	}
	cout << endl;
	trial_thr(a, n_all, n_model_inputs, outputs, ntarg, nhid_all, weigths_op, hid_act,
		final_layer_weigths, classifier);

	for (int i = 0; i < 2; i++)
		cout << outputs[i] << "  ";
}
void main()
{
	test2();
}
