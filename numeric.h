#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

template <class T>
using vec2D = vector< vector<T> >;

vec2D<float> XOR(vec2D<float> &x) {
  const int len = x.size();
  vec2D<float> y(len, vector<float>(1, 0.0));

  for(auto i = 0; i < len; i++) {
    vector<float> pair = x[i];
    y[i][0] = pair[0] * (1 - pair[1]) + pair[1] * (1 - pair[0]);
  }

  return y;
}

float sigmoid(float h, bool derv=false) {
	if(derv) {
		return h * (1 - h);
	}

  return 1 / (1 + exp(-h) );
}

float crossEntropy(float yhat, float y, bool derv=false) {
	if(derv) {
		return (yhat - y) / (yhat * (1- yhat) );
	}

	return -(y * log(yhat)) - ( (1 - y) * log(1 - yhat) );
}

vec2D<float> zeros(int m, int n) {
	return vec2D<float>(m, vector<float>(n, 0.0));
}

vec2D<float> zeros(vec2D<float> &a) {
	const int m = a.size();
	const int n = a[0].size();

	return zeros(m, n);
}

vec2D<float> ones(int m, int n) {
	return vec2D<float>(m, vector<float>(n, 1.0));
}

vec2D<float> ones(vec2D<float> &a) {
	const int m = a.size();
	const int n = a[0].size();

	return ones(m, n);
}

void printVec(vec2D<float> &vec) {
  for(auto outer : vec) {
    for(auto inner : outer) {
      cout << inner << " ";
    }
    cout << endl;
  }

  cout << endl;
}

vec2D<float> dot(vec2D<float> &a, vec2D<float> &b) {
	const int m = a.size();
	const int n = b[0].size();
	const int o = a[0].size();

	vec2D<float> c(m, vector<float>(n, 0.0));
	for(auto i = 0; i < m; i++) {
		for(auto j = 0; j < n; j++) {
			for(auto k = 0; k < o; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return c;
}

vec2D<float> multiply(vec2D<float> &a, vec2D<float> &b) {
	const int m = a.size();
	const int n = a[0].size();

	vec2D<float> c(m, vector<float>(n, 0.0));
	for(auto i = 0; i < m; i++) {
		for(auto j = 0; j < n; j++) {
			c[i][j] = a[i][j] * b[i][j];
		}
	}

	return c;
}

vec2D<float> multiply(vec2D<float> &a, float b) {
	const int m = a.size();
	const int n = a[0].size();

	vec2D<float> _b(a.size(), vector<float>(a[0].size(), b));

	return multiply(a, _b);
}

vec2D<float> divide(vec2D<float> &a, vec2D<float> &b) {
	const int m = a.size();
	const int n = a[0].size();

	vec2D<float> c(m, vector<float>(n, 0.0));
	for(auto i = 0; i < m; i++) {
		for(auto j = 0; j < n; j++) {
			c[i][j] = a[i][j] / b[i][j];
		}
	}

	return c;
}

vec2D<float> power(vec2D<float> &a, float k) {
	const int m = a.size();
	const int n = a[0].size();
	
	vec2D<float> c(m, vector<float>(n, 1.0));
	for(auto i = 0; i < m; i++) {
		for(auto j = 0; j < n; j++) {
			c[i][j] = pow(a[i][j], k);
		}
	}
	// for(auto i = 0; i < n; i++) {
	// 	c = multiply(c, a);
	// }

	return c;
}

vec2D<float> add(vec2D<float> &a, vec2D<float> &b) {
	const int m = a.size();
	const int n = a[0].size();

	vec2D<float> c(m, vector<float>(n, 0.0));
	for(auto i = 0; i < m; i++) {
		for(auto j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}

	return c;
}

float sumVec(vec2D<float> &a) {
	float _sum = 0.0;
	for(auto outer : a) {
		for(auto inner: outer) {
			_sum += inner;
		}
	}

	return _sum;
}

vec2D<float> subtract(vec2D<float> &a, vec2D<float> &b) {
	const int m = a.size();
	const int n = a[0].size();

	vec2D<float> c(m, vector<float>(n, 0.0));
	for(auto i = 0; i < m; i++) {
		for(auto j = 0; j < n; j++) {
			c[i][j] = a[i][j] - b[i][j];
		}
	}

	return c;
}

vec2D<float> transpose(vec2D<float> &a) {
	const int m = a.size();
	const int n = a[0].size();

	vec2D<float> c(n, vector<float>(m, 0.0));
	for(auto i = 0; i < m; i++) {
		for(auto j = 0; j < n; j++) {
			c[j][i] = a[i][j];
		}
	}

	return c;
}
