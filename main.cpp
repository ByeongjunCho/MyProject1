#include "Function.h"

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

	double w00[3] = { 0.2,-0.1,0 };
	double w01[3] = { 2,1,0 };
	int n_all = 2;
	int n_model_inputs = 2;
	double outputs[2];
	int ntarg = 2;
	int nhid_all[] = { 2 };
	double w1[6] = { 0.2,-0.1,0,2,1,0 };
	double *weigths_op[] = { w1 };

	double h1[2];
	double *hid_act[] = { h1 };
	double final_layer_weigths[] = { 0.1,0.2,-0.1,0.4,0 };
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
	double a[] = { 0,0,0,1,1,0,1,1 };  // 입력

	int i = 0;

	double w1[12] = { 1,1,1,1,1,1,1,1,1,1,1,1 }; // 첫번째 가중치
	double w2[9] = { 1,1,1,1,1,1,1,1,1 };  // 두번째 가중치

	int istart = 0;
	int istop = 7;
	int n_all = 3;
	int n_model_inputs = 2;
	double outputs[4];
	int ntarg = 4;
	int nhid_all[] = { 3,3 };
	double *weigths_op[] = { w1,w2 };   // 히든 레이어 주소를 가진 벡터

	double h1[3];  // 첫번째 히든 레이어 뉴런 벡터
	double h2[3];  // 두번째 히든 레이어 뉴런 벡터
	double *hid_act[] = { h1,h2 };     // 뉴런 개수 주소를 가진 벡터
	double final_layer_weigths[] = { 1,1,1,1,1,1,1,1,1,1,1,1 };  // 결과값에 있는 벡터
	int classifier = 1;  // sigmoid


	for (i; i < 8; i++) {
		printf("%4.5f ", a[i]);
	}
	cout << endl;
	trial_thr(a, n_all, n_model_inputs, outputs, ntarg, nhid_all, weigths_op, hid_act,
		final_layer_weigths, classifier);

	for (int i = 0; i < 4; i++)
		cout << outputs[i] << "  ";
}
void main()
{
	test2();
	int h1[3] = { 0,1,2 };  // 첫번째 히든 레이어 뉴런 벡터
	int h2[3] = { 0,1,2 };  // 두번째 히든 레이어 뉴런 벡터
	int *hid_act[] = { h1,h2 };     // 뉴런 개수 주소를 가진 벡터
	cout << hid_act[0] << endl;
	cout << hid_act[0] + 1 << endl;
	cout << hid_act[1]<< endl;
	cout << 1.0 / (1.0 + exp(-2)) << endl;
}
