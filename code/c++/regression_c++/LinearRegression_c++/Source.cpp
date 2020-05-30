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

		// 行列x_tildaの生成
		cv::Mat x_tilda(x_mat_train.rows, 1 + x_mat_train.cols, CV_64FC1);

		for (int i = 0; i < x_mat_train.rows; i++) {
			for (int j = 0; j < x_mat_train.cols + 1; j++) {
				if (j == 0){
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
		// 行列x_tildaの生成
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

int main(int argc, char* argv[])
{
	double x[][2] = { {33.0, 22.0},{31.0, 26.0},{32.0, 28.0} }; // 学習データの説明変数x
	double y[] = { 382.0, 324.0, 350 }; // 学習データの目的変数y

	Mat x_vector(3, 2, CV_64FC1, x);
	Mat y_vector(3, 1, CV_64FC1, y);

	LinearRegression model;

	//cout << x_vector.dot(y_vector) << endl;

	//cout << sum(y_vector)[0] << endl;

	//cout << x_vector.mul(x_vector) << endl;

	model.fit(x_vector, y_vector);
	cout << model.predict(x_vector) << endl;

	return 0;
}