#pragma once
#include <stdio.h>
#include <iostream>

using namespace std;

double dotprod(int n, double *vec1, double*vec2)  //(벡터 길이, 내적 연산에 사용될 벡터, 또 다른 벡터)
{
	int k, m;
	double sum = 0.0;

	k = n / 4; // 벡터를 4개의 그룹으로 나눈다.
	m = n % 4; // 나온 벡터의 개수를 저장한다.

	while (k--) { //4개로 나눠진 각 벡터 그룹마다 순환 루프 수행
		sum += *vec1* *vec2;
		//cout << sum << endl;
		sum += *(vec1 + 1) * *(vec2 + 1);
		//cout << sum << endl;
		sum += *(vec1 + 2) * *(vec2 + 2);
		//cout << sum << endl;
		sum += *(vec1 + 3) * *(vec2 + 3);
		//cout << sum << endl;
		//cout << "vec1 변경 전" << vec1 << endl;
		vec1 += 4;
		//cout << "vec1 : "<<vec1 << endl;
		vec2 += 4;
		//cout << "vec2: " <<vec2 << endl;
	}
	//cout << "벡터 순환문 끝" << endl;

	//cout << "m에 관한 순환문 시작" << endl;
	while (m--)
	{
		sum += *vec1++**vec2++;
		//cout << sum << endl;
	}

	return sum;
}


void activity(   // 뉴런 활성화 함수 => a=f(sum(weights) + bias) 
	double *input, // ninputs만큼의 길이를 갖는 현재 뉴런의 입력 벡터
	double *coefs, // ninputs + 1 길이를 갖는 가중치 벡터(바이어스를 마지막에 추가함 : {w1,w2, ..... ,wn,bias})
	double *output, // 도달한 현재 뉴런의 활성화(입력은 벡터)
	int ninputs,    // 입력 벡터의 길이(개수)
	int outlin = 0)     // 선형 여부 판단 변수(0 : 선형, 1 : sigmoid, 2 : ReLu)
{
	double sum = dotprod(ninputs, input, coefs);  // 입력과 weights의 내적
	sum += coefs[ninputs];  // 입력과 weights합에 bias를 더함.

	if (outlin == 0)
		*output = sum;   // 선형
	else if (outlin == 1)
		*output = 1.0 / (1.0 + exp(-sum)); // sigmoid
	else
	{
		if (sum > 0) *output = sum;
		else *output = 0;
	}
}
static void trial_thr(
	double *input,   //n_model_inputs만큼의 길이를 갖는 입력 벡터
	int n_all,		//출력은 포함하고 입력은 제외한 레이어의 개수
	int n_model_inputs, // 모델에 입력되는 입력의 개수
	double *outputs,    // ntarg만큼의 길이를 갖는 모델의 출력 벡터
	int ntarg,			// 모델의 최종 출력 개수
	int *nhid_all,		// nhid_all[i]은 i번째 은닉 레이어에 존재하는 은닉 뉴런의 개수
	double *weights_opt[],  // weights_opt[i]은 i번째 은닉 레이어의 가중치 벡터를 가리키는 포인터를 담고 있다.
	double *hid_act[],		// hid_act[i]는 i번째 은닉 레이어의 활성화 벡터를 가리키는 포인터를 담고 있다.
	double *final_layer_weights, //마지막 레이어의 가중치를 가리키는 포인터
	int classifier  // 0이 아니면 softmax출력, 1이면 선형 조합 출력
	)  
{
	int ilayer;  // 히든 레이어 수
	double sum;

	for (ilayer = 0; ilayer<n_all; ilayer++)   // 입력은 제외한 레이어부터 출력까지 레이어 선택
	{
		if (ilayer == 0 && n_all == 1)  //입력이 곧바로 출력되는 경우(은닉 레이어가 없는 경우)
		{
			for (int i = 0; i < ntarg; i++)   // 은닉레이어가 없기 때문에 바로 출력으로 계산(ntarg = 모델최종출력갯수)
				activity(input, final_layer_weights + i*(n_model_inputs + 1), outputs + i, n_model_inputs, 0);
			// (입력벡터, 마지막 레이어 벡터+i*(입력 개수+1), 출력벡터+i, 모델 입력 개수, 선형)
			// => activity(입력의 시작 포인터, 내적벡터의 시작 포인터, 출력벡터의 포인터, 입력벡터 길이, 선형여부)
			// 이 방법을 보면 가중치를 벡터로 길게 늘려서 사용하는것을 알 수 있다.
		}

		else if (ilayer == 0) // 첫 번째 은닉 레이어
		{
			for (int i = 0; i < nhid_all[ilayer]; i++) // 첫 번째 은닉 레이어에 존재하는 뉴런의 갯수 = nhid_all[0]
				activity(input, weights_opt[ilayer] + i*(n_model_inputs + 1), hid_act[ilayer] + i, n_model_inputs, 0);

			/* 입력 벡터
			weights_opt = {{0번째 레이어의 가중치 벡터},
			{1번째 레이어의 가중치 벡터},
			{2번째 레이어...},
			......}
			ilayer번째 가중치 벡터 + i번째 뉴런*(모델 입력 개수 + 1<bias 추가>)

			hid_act = {{00뉴런 값, 01뉴런 값, ...... 0x뉴런 값},
			{10뉴런 값, 11뉴런 값, ...... 0y뉴런 값},
			{........}}
			여기서 hid_act는{{0번째 레이어 뉴런의 값들},{1번째 레이어 뉴런의 값들},.....}
			의 모양으로 구성되어 있다.
			학습에서 가중치가 모인 행렬을 통해 계산을 진행한다. */

		}

		else if (ilayer < n_all - 1) // 중간 위치 은닉 레이어
		{
			for (int i = 0; i < nhid_all[ilayer]; i++)
				activity(hid_act[ilayer - 1], weights_opt[ilayer] + i*(nhid_all[ilayer - 1] + 1), hid_act[ilayer] + i, nhid_all[ilayer - 1], 0);
		}

		else  // 출력 레이어인 경우
		{
			for (int i = 0; i < ntarg; i++)
				activity(hid_act[ilayer - 1], final_layer_weights + i*(nhid_all[ilayer - 1] + 1), outputs + i, nhid_all[ilayer - 1], 1);
		}
	}

	if (classifier)   // 클래스 분류가 목적이면 항상 SoftMax를 이용한다.
	{
		sum = 0.0;
		for (int i = 0; i < ntarg; i++)   // 모델 출력을 처음부터 끝까지 for문으로 
		{
			if (outputs[i] < 300.0)
				outputs[i] = exp(outputs[i]);    // 모든 output을 exp함
			else
				outputs[i] = exp(300.0);
			sum += outputs[i];		// 모든 output값을 더해준다.
		}
		for (int i = 0; i < ntarg; i++)
			outputs[i] /= sum;     // outputs[i] = outputs / sum;
								   // 이 과정을 거쳐 확률값으로 변환한다.
	}
}

double batch_gradient(
	int istart,  //입력 행렬의 첫 번째 데이터 인덱스  
	int istop,   // 지난 마지막 데이터의 인덱스
	double *input, // 입력 행렬; 각 데이터의 길이 == max_neurons   => 입력 행렬은 벡터(행렬을 벡터로 변환한 것)
	double *targets, //목표 행렬; 각 데이터의 길이 == ntarg
	int n_all,   // 출력은 포함하고 입력은 제외한 레이어 개수
	int n_all_weights, // 마지막 레이어와 모든 바이어스 항을 포함한 총 가중치 개수
	int n_model_inputs, // 모델 입력의 개수; 입력 행렬은 더 많은 열을 가질 수도 있음
	double *outputs,    //모델 출력 벡터; 여기서는 작업 벡터로 사용됨
	int ntarg,   //출력 개수
	int *nhid_all, // nhid_all[i]은 i번째 은닉 레이어에 존재하는 뉴런의 개수
	double *weights_opt[], // weights_opt[i]는 i번째 은닉 레이어의 가중치 벡터를 가리키는 포인터
	double *hid_act[],   // hid_act[i]는 i번째 은닉 레이어의 활성화 벡터를 가리키는 포인터
	int max_neurons,   // 입력 행렬의 열의 개수; n_model_inputs보다 최대치가 크다
	double *this_delta, // 현재 레이어에 대한 델타 변수를 가리키는 포인터
	double *prior_delta, //다음 단계에 사용하기 위해 이전 레이어에서 미리 저장해놓은 델타 변수를 가리키는 포인터
	double **grad_ptr, // grad_ptr[i]는 i번째 레이어의 기울기를 가리키는 포인터
	double *final_layer_weights, // 마지막 레이어의 가중치를 가리키는 포인터
	double *grad,   // 계산된 모든 기울기로, 하나의 긴 벡터를 가리키는 포인터
	int classifier) // 0이 아니면 SoftMax결과 출력, 0이면 선형 결과 출력
					/* 모든 레이어를 거치면서 하나의 벡터 변수인 grad를 통해 총 n_all_weights개의 기울기 값들을 관리한다.
					grad_ptr은 입력 레이어를 제외한 전체 레이어의 수만큼 많은 수의 기울기 데이터를 관리한다.*/
{
	int i, j, icase, ilayer, nprev, nthis, nnext, imax;
	double diff, *dptr, error, *targ_ptr, *prevact, *gradptr, delta, *nextcoefs, tmax;

	for (i = 0; i < n_all_weights; i++)  // 기울기 0으로 초기화
		grad[i] = 0.0;    // 모든 레이어의 기울기 0으로 초기화.

	error = 0.0;      // 전체 오차 값을 누적해나간다.
	for (icase = istart; icase < istop; icase++)  // 입력 행렬의 첫 번째 데이터 인덱스 ~ 마지막 데이터 인덱스
	{
		dptr = input + icase*max_neurons;   // 현재 데이터를 가리킨다. 행렬에서 icase번째 행을 선택하는 것과 같다.
											// 입력은 벡터이므로 열 개수를 기준으로 입력을 정한다.
		trial_thr(dptr, n_all, n_model_inputs, outputs, ntarg, nhid_all, weights_opt, hid_act, final_layer_weights, classifier);

		targ_ptr = targets + icase*ntarg; // 현재 훈련 벡터에 대응하는 타겟 벡터를 가리키는 포인터
										  // 정답값


		if (classifier)   // Softmax를 사용한 경우
		{
			tmax = -1.e30;
			for (i = 0; i < ntarg; i++)   // 최대값을 갖는 참 클래스를 찾는다.
			{
				if (targ_ptr[i] > tmax)
				{
					imax = i;				// imax에 softmax최대값이 있는 인덱스 저장
					tmax = targ_ptr[i];
				}
				this_delta[i] = targ_ptr[i] - outputs[i]; //교차 엔트로피를 입력으로 미분해 음의 부호를 취한 식
														  // 출력값과 정답값의 차이를 this_delta벡터에 저장한다.
			}
			error -= log(outputs[imax] + 1.e-30); // error = error-log(outputs[imax]+1.e-30) 음의 로그 확률을 최소화한다.
		}
		else
		{
			for (i = 0; i < ntarg; i++)
			{
				diff = outputs[i] - targ_ptr[i];
				error += diff*diff;   // error = error + diff^2
				this_delta[i] = -2.0*diff;
			}
		}

		if (n_all == 1)    // 은닉 레이어가 없는 경우
		{
			nprev = n_model_inputs;   // 출력 레이어에 전달되는 입력의 개수
			prevact = input + icase*max_neurons; //현재 데이터를 가리키는 포인터
		}
		else
		{
			nprev = nhid_all[n_all - 2];  // n_all-2인덱스가 마지막 은닉 레이어의 개수가 있는 주소이다.
			prevact = hid_act[n_all - 2]; // 출력 레이어로 전달되는 각각의 뉴런의 포인터 변수
		}
		gradptr = grad_ptr[n_all - 1];  // 기울기 벡터에서 출력레이어의 기울기를 가리키는 포인터

		for (i = 0; i < ntarg; i++)
		{
			delta = this_delta[i];  //평가 기준을 logit으로 편미분해 음수를 취한다.
			for (j = 0; j < nprev; j++)
				*gradptr++ += delta*prevact[j];  //모든 훈련 데이터에 대한 결과를 누적한다.

			*gradptr++ += delta;  // 바이어스 활성화는 항상 1이다. 
		}
		nnext = ntarg;   // 레이어 되돌아갈 준비를 한다.
		nextcoefs = final_layer_weights;

		for (ilayer = n_all - 2; ilayer >= 0; ilayer--) {   // 각 은닉 레이어마다 역방향으로 진행해 나간다.
			nthis = nhid_all[ilayer];        // 현재 은닉 레이어상에 존재하는 뉴런의 개수
			gradptr = grad_ptr[ilayer];      // 현재 레이어의 기울기를 가리키는 포인터
			for (i = 0; i<nthis; i++) {       // 현재 레이어상의 뉴런들에 대해 루프 수행
				delta = 0.0;
				for (j = 0; j<nnext; j++)
					delta += this_delta[j] * nextcoefs[j*(nthis + 1) + i];
				delta *= hid_act[ilayer][i] * (1.0 - hid_act[ilayer][i]);  // 미분 연산
				prior_delta[i] = delta;                    // 다음 레이어를 위해 저장
				if (ilayer == 0) {                          // 첫 번째 히든 레이어인가?
					prevact = input + icase * max_neurons;  // Point to this sample
					for (j = 0; j<n_model_inputs; j++)
						*gradptr++ += delta * prevact[j];
				}
				else {      // 적어도 하나 이상의 은닉 레이어가 현재 레이어 이전에 존재
					prevact = hid_act[ilayer - 1];
					for (j = 0; j<nhid_all[ilayer - 1]; j++)
						*gradptr++ += delta * prevact[j];
				}
				*gradptr++ += delta;   // 바이어스 활성화는 언제나 1이다.
			}  // 현재 모든 뉴런을 대상으로 한다.

			for (i = 0; i<nthis; i++)           // 현재 델타 값을 이전 델타 값으로 저장
				this_delta[i] = prior_delta[i];

			nnext = nhid_all[ilayer];          // 다음 레이어를 위한 준비
			nextcoefs = weights_opt[ilayer];
		}  // 모든 레이어를 대상으로 거꾸로 실행해 나간다.

	} // MSE나 음의 로그 발생 가능 확률 반환

	return error;  // MSE or negative log likelihood
}
