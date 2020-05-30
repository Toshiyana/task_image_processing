#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

class SimpleRegression
{
private:
	int size;

public:
	double intercept;
	double coefficient;

	void fit(cv::Mat x_vec_train, cv::Mat y_vec_train) {
		this->size = x_vec_train.rows;
		this->coefficient = (x_vec_train.dot(y_vec_train) - sum(x_vec_train)[0] * sum(y_vec_train)[0] / this->size) / (sum(x_vec_train.mul(x_vec_train))[0] - sum(x_vec_train)[0] * sum(x_vec_train)[0] / this->size);
		this->intercept = (sum(y_vec_train)[0] - this->coefficient * sum(x_vec_train)[0]) / this->size;
	}

	cv::Mat predict(cv::Mat x_vec_test) {
		return this->coefficient * x_vec_test + this->intercept;
	}
};

int main(int argc, char* argv[])
{
	double x[] = { 1, 2, 3, 6, 7, 9 }; // 学習データの説明変数x
	double y[] = { 1, 3, 3, 5, 4, 6 }; // 学習データの目的変数y

	Mat x_vector(6, 1, CV_64FC1, x);
	Mat y_vector(6, 1, CV_64FC1, y);

	SimpleRegression model;

	//cout << x_vector.dot(y_vector) << endl;
	
	//cout << sum(y_vector)[0] << endl;

	//cout << x_vector.mul(x_vector) << endl;

	model.fit(x_vector, y_vector);
	cout << model.predict(x_vector) << endl;

	return 0;
}