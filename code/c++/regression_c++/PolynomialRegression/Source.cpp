#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

class LinearRegression
{
public:
	cv::Mat weight;

	void fit(cv::Mat x_mat_train, cv::Mat y_mat_train) {

		// �s��x_tilda�̐���
		cv::Mat x_tilda(x_mat_train.rows, 1 + x_mat_train.cols, CV_64FC1);

		for (int i = 0; i < x_mat_train.rows; i++) {
			for (int j = 0; j < x_mat_train.cols + 1; j++) {
				if (j == 0) {
					x_tilda.at<double>(i, j) = 1;
				}
				else {
					x_tilda.at<double>(i, j) = x_mat_train.at<double>(i, j - 1);
				}
			}
		}


		cv::Mat A = x_tilda.t() * x_tilda;
		cv::Mat B = x_tilda.t() * y_mat_train;
		this->weight = A.inv() * B;
	}

	cv::Mat predict(cv::Mat x_mat_test) {
		// �s��x_tilda�̐���
		cv::Mat x_tilda(x_mat_test.rows, 1 + x_mat_test.cols, CV_64FC1);

		for (int i = 0; i < x_mat_test.rows; i++) {
			for (int j = 0; j < x_mat_test.cols + 1; j++) {
				if (j == 0) {
					x_tilda.at<double>(i, j) = 1;
				}
				else {
					x_tilda.at<double>(i, j) = x_mat_test.at<double>(i, j - 1);
				}
			}
		}

		return x_tilda * this->weight;
	}

};

// �����ϐ�����̎��̂�
class PolynomialRegression
{
public:
	cv::Mat weight;
	int degree; //�ߎ�����������
	cv::Mat w; //�d��

	void fit(cv::Mat x_mat_train, cv::Mat y_mat_train, int deg) {
		this->degree = deg;

		// �w�K�f�[�^�ɂ��s��M�̐���(1�̒萔�����܂܂Ȃ�)
		cv::Mat M(x_mat_train.rows, this->degree, CV_64FC1);

		for (int i = 0; i < x_mat_train.rows; i++) {
			for (int j = 0; j < this->degree; j++) {
				M.at<double>(i, j) = pow(x_mat_train.at<double>(i, 0), j + 1);
			}
		}

		LinearRegression lr;
		lr.fit(M, y_mat_train);
		this->w = lr.weight;
	}

	cv::Mat predict(cv::Mat x_mat_test) {
		//���؃f�[�^�ɂ��s��M�̐����i1�̒萔�����܂ށj
		cv::Mat M(x_mat_test.rows, 1 + this->degree, CV_64FC1);

		for (int i = 0; i < x_mat_test.rows; i++) {
			for (int j = 0; j < this->degree + 1; j++) {
				M.at<double>(i, j) = pow(x_mat_test.at<double>(i, 0), j);
			}
		}

		return M * this->w;

	}

};


int main(int argc, char* argv[])
{
	double x[] = { 1, 2, 3, 6, 7, 9 }; // �w�K�f�[�^�̐����ϐ�x
	double y[] = { 1, 3, 3, 5, 4, 6 }; // �w�K�f�[�^�̖ړI�ϐ�y

	Mat x_vector(6, 1, CV_64FC1, x);
	Mat y_vector(6, 1, CV_64FC1, y);

	PolynomialRegression model;

	model.fit(x_vector, y_vector, 3);

	cout << model.predict(x_vector);

	//cout << x_vector.dot(y_vector) << endl;

	//cout << sum(y_vector)[0] << endl;

	//cout << x_vector.mul(x_vector) << endl;

	//model.fit(x_vector, y_vector);
	//cout << model.predict(x_vector) << endl;

	return 0;
}