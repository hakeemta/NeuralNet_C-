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

vec2D<float> power(vec2D<float> &a, int n) {
	vec2D<float> c(a.size(), vector<float>(a[0].size(), 1.0));
	for(auto i = 0; i < n; i++) {
		c = multiply(c, a);
	}

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

class Layer {
  public:
		vec2D<float> weights;

    // Default constructor
    Layer() {
      ;
    }

    Layer(int _nInput, int _nOutput, float (*_actFn)(float, bool) ) {
      nInput = _nInput;
      nOutput = _nOutput;
      actFn = _actFn;

      init();

    }

  void init() {
    weights.assign(
      nInput, 
      vector<float> (nOutput, 0.0)
    );

    // For each of the layer inputs
    for(int i = 0; i < nInput; i++) {
      // For each of the layer outputs
      for(int j = 0; j < nOutput; j++) {
        float val = (rand() % 10000) / 10000.0;
        weights[i][j] = val;
      }
    }

    bias.assign(1, vector<float>(nOutput, 0.0));

    cout << "W: " << endl;
    printVec(weights);

    cout << "b: " << endl;
    printVec(bias);
  }

  vec2D<float> forward(vec2D<float> &_X) {
		X.assign(_X.size(), vector<float>(_X[0].size(), 0.0));
		for(auto i = 0; i < _X.size(); i++) {
			X[i] = vector<float>(_X[i]);
		}

    // h = X.W + b
    vec2D<float> h;
    h = dot(X, weights);
    
    vec2D<float> _b(h.size(), bias[0]);
    h = add(h, _b);

		y.assign(h.size(), vector<float>(h[0].size(), 0.0) );
		for(auto i = 0; i < h.size(); i++){
			for(auto j = 0; j < h[0].size(); j++) {
				y[i][j] = actFn(h[i][j], false);
			}
		}

		return y;
  }

	vec2D<float> backprop(vec2D<float> dEdy) {
		vec2D<float> actDervY(y.size(), vector<float>(y[0].size(), 0.0) );
		for(auto i = 0; i < y.size(); i++){
			for(auto j = 0; j < y[0].size(); j++) {
				actDervY[i][j] = actFn(y[i][j], true);
			}
		}

		vec2D<float> delta = multiply(dEdy, actDervY);

		vec2D<float> dw = transpose(X);
		dw = dot(dw, delta);

		vec2D<float> id(1, vector<float>(delta.size(), 1.0) );
		vec2D<float> db = dot(id, delta);

		float lr = 0.01;
		
		dw = multiply(dw, lr);
		weights = subtract(weights, dw);
		// printVec(weights);

		db = multiply(db, lr);
		bias = subtract(bias, db);
		// printVec(bias);

		return delta;
	}

  private:
    int nInput;
    int nOutput;
    float (*actFn) (float, bool);
    vec2D<float> bias;
    vec2D<float> X;
		vec2D<float> y;
};

class NN {
  public:
    NN(int _nInput, int _nHidden, int _nOutput) {
      nInput = _nInput;
      nHidden = _nHidden;
      nOutput = _nOutput;

      hiddenLayer = Layer(nInput, nHidden, sigmoid);
      outputLayer = Layer(nHidden, nOutput, sigmoid);
    }

    vec2D<float> forward(vec2D<float> X) {
      vec2D<float> hiddenY = hiddenLayer.forward(X);
      vec2D<float> outputY = outputLayer.forward(hiddenY);		

      return outputY;
    }

		float backprop(vec2D<float> yhat, vec2D<float> y) {
			// Compute error
			vec2D<float> vec = subtract(yhat, y);
			vec = power(vec, 2);
			float error = 0.5 * sumVec(vec) / yhat.size();

			vec2D<float> dEdy = subtract(yhat, y);

			// Output backprop
			vec2D<float> outputDelta = outputLayer.backprop(dEdy);

			// Hidden contributory errors; then hidden backprop
			vec2D<float> _w = transpose(hiddenLayer.weights);
			vec2D<float> hiddenError = dot(outputDelta, _w);
			vec2D<float> hiddenDelta = hiddenLayer.backprop(hiddenError);
			
			return error;
		}

		vec2D<float> fit(vec2D<float> X, vec2D<float> y, long epochs, bool verbose=false) {
			vec2D<float> loss;

			for(auto i = 0; i < epochs; i++) {
				// Run forward
				vec2D<float> yhat = forward(X);

				// Run backprop
				float error = backprop(yhat, y);

				
			}

			return loss;
		}

		vec2D<float> predict(vec2D<float> X) {
			// Run forward
			vec2D<float> yhat = forward(X);

			return yhat;
		}

  private:
    int nInput;
    int nHidden;
    int nOutput;

    Layer hiddenLayer;
    Layer outputLayer;
};

int main()
{
  // Input X
  vec2D<float> X = {
    vector<float> {0.0, 0.0},
    vector<float> {0.0, 1.0},
    vector<float> {1.0, 0.0},
    vector<float> {1.0, 1.0}
  };
  //cout << "X:" << endl;
	//printVec(X);
	
  vec2D<float> y = XOR(X);
  //cout << "y:" << endl;
  //printVec(y);

  NN model = NN(2, 2, 1);
  model.fit(X, y, 100000);

	vec2D<float> yhat = model.predict(X);

	cout << "Pred:" << endl;
	printVec(yhat);
	
}
