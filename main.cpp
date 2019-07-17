#include <iostream>
#include <vector>
#include <math.h>
#include "numeric.h"

using namespace std;

enum optimizer_t {SGD, RMS_PROP};
enum loss_t {MSE, CE};

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

			lr = 0.01;
			beta = 0.9;
			eps = 1e-8;

      init();

    }

  void init() {
		weights = zeros(nInput, nOutput);
    // For each of the layer inputs
    for(int i = 0; i < nInput; i++) {
      // For each of the layer outputs
      for(int j = 0; j < nOutput; j++) {
        float val = (rand() % 10000) / 10000.0;
        weights[i][j] = val;
      }
    }

		bias = zeros(1, nOutput);

    cout << "W: " << endl;
    printVec(weights);

    cout << "b: " << endl;
    printVec(bias);

		// RMSprop optimizer momentum params
		vW = zeros(weights);
		vB = zeros(bias);
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

	vec2D<float> movingAvg(vec2D<float> v, vec2D<float> d) {
		d = power(d, 2);
		d = multiply(d, beta);
		
		v = multiply(v, (1 - beta));
		v = add(v, d);

		return v;
	}

	vec2D<float> momentum(vec2D<float> d, vec2D<float> v) {
		v = power(v, 0.5);
		vec2D<float> _eps = ones(v);
		_eps = multiply(_eps, eps);
		v = add(v, _eps);

		return divide(d, v);
	}


	vec2D<float> backprop(vec2D<float> dEdy, optimizer_t opt=SGD) {
		vec2D<float> actDervY(y.size(), vector<float>(y[0].size(), 0.0) );
		for(auto i = 0; i < y.size(); i++){
			for(auto j = 0; j < y[0].size(); j++) {
				actDervY[i][j] = actFn(y[i][j], true);
			}
		}

		vec2D<float> delta = multiply(dEdy, actDervY);

		vec2D<float> dW = transpose(X);
		dW = dot(dW, delta);

		vec2D<float> id(1, vector<float>(delta.size(), 1.0) );
		vec2D<float> dB = dot(id, delta);

		dW = multiply(dW, lr);
		dB = multiply(dB, lr);

		if(opt == RMS_PROP){
			// RMSprop optimizer
			vW = movingAvg(vW, dW);
			vB = movingAvg(vB, dB);

			dW = momentum(dW, vW);
			dB = momentum(dB, vB);
		}
		
		weights = subtract(weights, dW);
		// printVec(weights);
		bias = subtract(bias, dB);
		// printVec(bias);

		return delta;
	}

  private:
    int nInput;
    int nOutput;
    float (*actFn) (float, bool);
    vec2D<float> bias;
		vec2D<float> vW;
		vec2D<float> vB;
    vec2D<float> X;
		vec2D<float> y;
		float lr;
		float beta;
		float eps;
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

		float backprop(vec2D<float> yhat, vec2D<float> y, loss_t lossFn=MSE, optimizer_t opt=SGD) {
			// Compute error
			// vec2D<float> vec = subtract(yhat, y);
			// vec = power(vec, 2);
			// float error = 0.5 * sumVec(vec) / yhat.size();
			// vec2D<float> dEdy = subtract(yhat, y);

			vec2D<float> vec;
			float error;
			vec2D<float> dEdy;
			if(lossFn == MSE) {
				vec = subtract(yhat, y);
				vec = power(vec, 2);
				error = 0.5 * sumVec(vec) / yhat.size();
				
				dEdy = subtract(yhat, y);
			}
			else if(lossFn == CE) {
				vec = zeros(y);
				dEdy = zeros(y);
				for(auto i = 0; i < y.size(); i++){
					for(auto j = 0; j < y[0].size(); j++) {
						vec[i][j] = crossEntropy(yhat[i][j], y[i][j]);

						dEdy[i][j] = crossEntropy(yhat[i][j], y[i][j], true);
					}
				}
				error = sumVec(vec) / yhat.size();

				for(auto i = 0; i < y.size(); i++){
					for(auto j = 0; j < y[0].size(); j++) {
						vec[i][j] = crossEntropy(yhat[i][j], y[i][j]);
					}
				}
			}

			// Output backprop
			vec2D<float> outputDelta = outputLayer.backprop(dEdy, opt);

			// Hidden contributory errors; then hidden backprop
			vec2D<float> _w = transpose(hiddenLayer.weights);
			vec2D<float> hiddenError = dot(outputDelta, _w);
			vec2D<float> hiddenDelta = hiddenLayer.backprop(hiddenError);
			
			return error;
		}

		vector<float> fit(vec2D<float> X, vec2D<float> y, long epochs, bool verbose=false, loss_t lossFn=MSE, optimizer_t opt=SGD) {
			vector<float> errors;

			for(auto i = 0; i < epochs; i++) {
				// Run forward
				vec2D<float> yhat = forward(X);

				// Run backprop
				float error = backprop(yhat, y, lossFn, opt);

				errors.push_back(error);
				if(verbose && i % 1000 == 0) {
					cout << error << endl;
				}	
			}

			return errors;
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

	// srand(1);

  NN model = NN(2, 2, 1);
  vector<float> loss = model.fit(X, y, 8000, true, CE, RMS_PROP);

	vec2D<float> yhat = model.predict(X);

	cout << "Pred:" << endl;
	printVec(yhat);
	
}
