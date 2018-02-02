#include "Vector.h"
void main()
{
	double b[4] = { 1, 2, 5, 7 };
	double c[4] = { 1, 4, 8, 2 };
	double result;
	double a[] = {0,0,0,1,1,0,1,1};

	int i;
	i = 0;

	int n_all = 1;
	int n_model_inputs = 2;
	double outputs[4];
	int ntarg = 4;
	int *nhid_all = {};
	double *weigths_op;
	double *hid_act;
	double final_layer_weigths[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	int classifier = 1;


	for (i; i < 8; i++) {
		printf("%4.5f", a[i]);
	}
	result = dotprod(4, c, b);
	printf("\n This is the result : %4.5f \n ", result);
	
	trial_thr(a, n_all, n_model_inputs, outputs, ntarg, nhid_all, &weigths_op, &hid_act,
		final_layer_weigths, classifier);

	for (int i = 0; i < 4; i++)
		cout << outputs[i] << "  ";
}
