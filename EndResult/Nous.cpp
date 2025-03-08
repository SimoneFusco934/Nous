#include <random>
#include <cstring>

namespace nous {
	float ***weights;
	float **biases;
	float **z;
	float **a;
	float ***dw;
  float **db;
	float **dz;
	float **da;
	float *c;
	int *weightsRowsMapping;
	int *weightsColumnsMapping;
	int *bzaRowsMapping;
	int *activation_functions;
	int loss_function_index;
	int layers;
	float learning_rate;
	int batch_counter = 0;
	int batch_size;
}

//eg. layers = 3 (2 hidden and 1 output), neuronsMapping = [784, 25, 16, 10], activation_functions = [0, 0, 1], loss_function_index = 1, learning_rate 0.01.
void initializeNN(int layers, int *neuronsMapping, int *activation_functions, int loss_function_index, float learning_rate, int batch_size){

	nous::layers = layers;	
	nous::activation_functions = activation_functions;
	nous::loss_function_index = loss_function_index;
	nous::learning_rate = learning_rate;
	nous::batch_size = batch_size;

	nous::weightsRowsMapping = new int[layers];
	nous::weightsColumnsMapping = new int[layers];
	nous::bzaRowsMapping = new int[layers];

	//Initializing weights and dw arrays
	nous::weights = new float **[layers];
	nous::dw = new float **[layers];

  for(int i = 0; i < layers; i++){
		nous::weights[i] = new float *[neuronsMapping[i + 1]];
		nous::dw[i] = new float *[neuronsMapping[i + 1]];
		nous::weightsRowsMapping[i] = neuronsMapping[i + 1];
  }

  for(int i = 0; i < layers; i++){
    for(int j = 0; j < nous::weightsRowsMapping[i]; j++){
			nous::weights[i][j] = new float[neuronsMapping[i]];
			nous::dw[i][j] = new float[neuronsMapping[i]];
			nous::weightsColumnsMapping[i] = neuronsMapping[i];
    }
  }

	//Declaring biases, z, a, dz, da arrays
	nous::biases = new float *[layers];
	nous::z = new float *[layers];
	nous::a = new float *[layers];
	nous::db = new float *[layers];
	nous::dz = new float *[layers];
	nous::da = new float *[layers];

  for(int i = 0; i < layers; i++){
		nous::biases[i] = new float[neuronsMapping[i + 1]];
		nous::db[i] = new float[neuronsMapping[i + 1]];
		nous::z[i] = new float[neuronsMapping[i + 1]];
		nous::a[i] = new float[neuronsMapping[i + 1]];
		nous::dz[i] = new float[neuronsMapping[i + 1]];
		nous::da[i] = new float[neuronsMapping[i + 1]];
		nous::bzaRowsMapping[i] = neuronsMapping[i + 1];
  }

  //Setting db, dz and da to 0
  for(int i = 0; i < layers; i++){
		memset(nous::db[i], 0, nous::bzaRowsMapping[i]);
		memset(nous::dz[i], 0, nous::bzaRowsMapping[i] * sizeof(float));
		memset(nous::da[i], 0, nous::bzaRowsMapping[i] * sizeof(float));
  }

	//Setting dw to 0
	for(int i = 0; i < layers; i++){
		for(int j = 0; j < nous::weightsRowsMapping[i]; j++){
			memset(nous::dw[i][j], 0, nous::weightsColumnsMapping[i]);
		}
	}

	//Declaring c array
	nous::c = new float[neuronsMapping[layers]];
}



/* MATRIX OPERATIONS */
//Matrix multiplication between 2d matrix (rows:m x columns:n) and 1d array (rows:n x columns:1)
void matrixMultiplication(float **twodArray, float *onedArray, float *resultArray, int m, int n){
  for(int i = 0; i < m; i++){
    float sum = 0;
    for(int j = 0; j < n; j++){
      sum += twodArray[i][j] * onedArray[j];
    }
    resultArray[i] = sum;
  }
}

//Matrix sum between 1d array (rows:m x columns:1) and 1d array (rows:m x columns: 1)
void matrixSum(float *onedArray, float *onedArray2, float *resultArray, int m){
	for(int i = 0; i < m; i++){
		resultArray[i] = onedArray[i] + onedArray2[i];
	}
}

//Relu activation function over 1d array (rows:m x columns:1)
void matrixRelu(float *onedArray, float *resultArray, int m){
	for(int i = 0; i < m; i++){
		resultArray[i] = relu(onedArray[i]);
	}
}



/* WEIGHTS AND BIASES INITIALIZATION */
//Weights initialization. Supported initialization techniques: He Normal (0), Xavier Normal (1).
void initializeWeights(int *initialization_technique){

	using namespace nous;

  random_device rd;
  mt19937 gen(rd());

  for(int i = 0; i < layers; i++){
    if(initialization_technique[i] == 0){
      //He Normal
      for(int j = 0; j < weightsRowsMapping[i]; j++){
        for(int h = 0; h < weightsColumnsMapping[i]; h++){
          float sigma = sqrt(2 / float(weightsColumnsMapping[i]));
          normal_distribution<float> d(0,sigma);
          weights[i][j][h] = d(gen);
        }
      }
    }else if(initialization_technique[i] == 1){
      //Xavier Normal
      for(int j = 0; j < weightsRowsMapping[i]; j++){
        for(int h = 0; h < weightsColumnsMapping[i]; h++){
          float sigma = sqrt(6 / float(weightsColumnsMapping[i] + weightsRowsMapping[i]));
          normal_distribution<float> d(0,sigma);
          weights[i][j][h] = d(gen);
        }
      }
    }
  }
}

//Weights initialization with saved.txt file
void initializeWeightsSaved(std::ifstream& inFile){

	using namespace nous;

	for(int i = 0; i < layers; i++){
  	for(int j = 0; j < weightsRowsMapping[i]; j++){
    	for(int h = 0; h < weightsColumnsMapping[i]; h++){
      	inFile >> weights[i][j][h];				
      }
    }
  }

	//std::cout << "Last weight: " << weights[layers-1][weightsRowsMapping[layers-1]-1][weightsColumnsMapping[layers-1]-1] << std::endl;
}

//Biases initialization. Uses normal distribution with mean 0 and standard deviation 1.
void initializeBiases(){

	using namespace nous;

  random_device rd;
  mt19937 gen(rd());

  for(int i = 0; i < layers; i++){
    for(int j = 0; j < bzaRowsMapping[i]; j++){
      normal_distribution<float> d(0, 1);
      biases[i][j] = d(gen);
    }
  }
}

//Biases initialization with saved.txt file
void initializeBiasesSaved(std::ifstream& inFile){

	using namespace nous;

	for(int i = 0; i < layers; i++){
  	for(int j = 0; j < bzaRowsMapping[i]; j++){
    	inFile >>  biases[i][j];
    }
  }

	//cout << "last bias: " << biases[layers-1][bzaRowsMapping[layers-1]-1] << endl;
}



/* ACTIVATION FUNCTIONS AND THEIR DERIVATIVES */
//Relu activation function.
float relu(float x){
  if(x > 0){
    return x;
  }
  return 0;
}

//Softmax activation function.
void softmax(float *onedArray, float *resultArray, int size){

	float sum = 0;

	for(int i = 0; i < size; i++){
  	sum += exp(onedArray[i]);
  }


	for(int i = 0; i < size; i++){
		resultArray[i] = exp(onedArray[i]) / sum;
	}	
}

//Relu derivative.
float reluD(float x){
	if(x > 0){
		return 1;
	}

	return 0;
}

//Softmax derivative.
float softMaxD(float *a, int i, int j){

	float result = 0;

	if(i == j){
		result = a[j] * (1 - a[j]);		
	}else{
		result = - (a[j] * a[i]);
	}

	return result;
}



/* FEED FORWARD */
//Calculates z value for each neuron in layer: [z_layer].
void calculateZ(float *inputs, int z_layer){

	using namespace nous;
  
  if(z_layer == 0){
    matrixMultiplication(weights[z_layer], inputs, z[z_layer], weightsRowsMapping[z_layer], weightsColumnsMapping[z_layer]);
  }else{
    matrixMultiplication(weights[z_layer], a[z_layer - 1], z[z_layer], weightsRowsMapping[z_layer], weightsColumnsMapping[z_layer]);
  }

  matrixSum(z[z_layer], biases[z_layer], z[z_layer], bzaRowsMapping[z_layer]);
}

//Calculates a value for each neuron in layer: [a_layer]. Supported activation functions: relu (0), softmax (1).
void calculateA(int a_layer, int function_index){
	
	using namespace nous;

  if(function_index == 0){
    matrixRelu(z[a_layer], a[a_layer], bzaRowsMapping[a_layer]);
  }else if(function_index == 1){
    softmax(z[a_layer], a[a_layer], bzaRowsMapping[a_layer]);
  }
}

//Calculates c values for each neuron in the last layer. Supported loss functions: MSE (0), Cross Entropy Loss (1).
void calculateC(float *y, float &cost, int loss_function_index){

	using namespace nous;

  if(loss_function_index == 0){
    for(int i = 0; i < bzaRowsMapping[layers - 1]; i++){
      c[i] = (y[i] - a[layers - 1][i]) * (y[i] - a[layers - 1][i]);
    }
  }else if(loss_function_index == 1){
    for(int i = 0; i < bzaRowsMapping[layers - 1]; i++){
      c[i] = - (y[i] * log(a[layers - 1][i]));
    }
  }

  float sum = 0;

  for(int i = 0; i < bzaRowsMapping[layers - 1]; i++){
    sum += c[i];
  }

  cost = sum;
}



/* FEED BACKWARD */
//Calculates loss derivative with respect to each a value in the last layer. Supported loss functions: MSE (0), Cross Entropy Loss (1).
void calculateDLossToA(float *y, int index_loss_function){

	using namespace nous;

  if(index_loss_function == 0){
    for(int i = 0; i < bzaRowsMapping[layers - 1]; i++){
      da[layers - 1][i] += 2 * (y[i] - a[layers - 1][i]);
    }
  }else if(index_loss_function == 1){
    for(int i = 0; i < bzaRowsMapping[layers - 1]; i++){
      da[layers - 1][i] += - (y[i] / a[layers - 1][i]);
    }
  }
}

//Calculates a derivative with respect to each z value in layer: [index_layer]. Supported activation functions: relu (0), softmax (1).
void calculateDAToZ(int index_activation_function, int index_layer){

	using namespace nous;

  if(index_activation_function == 0){
    for(int i = 0; i < bzaRowsMapping[index_layer]; i++){
      dz[index_layer][i] += reluD(z[index_layer][i]) * da[index_layer][i];
    }     
  }else if(index_activation_function == 1){
    for(int i = 0; i < bzaRowsMapping[index_layer]; i++){
      for(int j = 0; j < bzaRowsMapping[index_layer]; j++){
        dz[index_layer][i] += softMaxD(a[index_layer], i, j) * da[index_layer][j];
      }
    }
  }
}

//Calculates z derivative with respect to each a value in layer: [index_layer].
void calculateDZToA(int index_layer){

	using namespace nous;

  for(int i = 0; i < weightsRowsMapping[index_layer + 1]; i++){
    for(int j = 0; j < weightsColumnsMapping[index_layer + 1]; j++){
      da[index_layer][j] += weights[index_layer + 1][i][j] * dz[index_layer + 1][i];
    }
  }
}

//Calculates z derivative with respect to each weight in layer: [index_layer], than updates the weight.
void calculateDZToWStochastic(float *inputs, int index_layer){

	using namespace nous;

  if(index_layer > 1){
    for(int i = 0; i < weightsRowsMapping[index_layer]; i++){
      for(int j = 0; j < weightsColumnsMapping[index_layer]; j++){
        weights[index_layer][i][j] -= a[index_layer - 1][j] * dz[index_layer][i] * learning_rate;
      }
    }
  }else{
    for(int i = 0; i < weightsRowsMapping[index_layer]; i++){
      for(int j = 0; j < weightsColumnsMapping[index_layer]; j++){
        weights[index_layer][i][j] -= inputs[j] * dz[index_layer][i] * learning_rate;
      }
    }
  }
}

//Calculates z derivative with respect to each bias in layer: [index_layer], then updates the bias.
void calculateDZToBStochastic(int index_layer){

	using namespace nous;

  for(int i = 0; i < bzaRowsMapping[index_layer]; i++){
    biases[index_layer][i] -= dz[index_layer][i] * learning_rate;
  }
}

void calculateDZToW(float *inputs, int index_layer){

  using namespace nous;

  if(index_layer > 1){
    for(int i = 0; i < weightsRowsMapping[index_layer]; i++){
      for(int j = 0; j < weightsColumnsMapping[index_layer]; j++){
        dw[index_layer][i][j] += a[index_layer - 1][j] * dz[index_layer][i];
      }
    }
  }else{
    for(int i = 0; i < weightsRowsMapping[index_layer]; i++){
      for(int j = 0; j < weightsColumnsMapping[index_layer]; j++){
        dw[index_layer][i][j] += inputs[j] * dz[index_layer][i];
      }
    }
  }
}

//Calculates z derivative with respect to each bias in layer: [index_layer].
void calculateDZToB(int index_layer){

  using namespace nous;

  for(int i = 0; i < bzaRowsMapping[index_layer]; i++){
    db[index_layer][i] += dz[index_layer][i];
  }
}




/* WEIGHTS AND BIASES UPDATES */
//Updates weights.

void updateW(){

  using namespace nous;

	for(int i = 0; i < layers; i++){
    for(int j = 0; j < weightsRowsMapping[i]; j++){
      for(int h = 0; h < weightsColumnsMapping[i]; h++){
        weights[i][j][h] -= dw[i][j][h] * learning_rate;
      }
    }
	}
}

//Updates biases.
void updateB(){

  using namespace nous;
	
	for(int i = 0; i < layers; i++){
  	for(int j = 0; j < bzaRowsMapping[j]; j++){
    	biases[i][j] -= db[i][j] * learning_rate;
  	}
	}
}



/* PROPAGATION */
//Forward Propagation. Returns an array containing each a value of the last layer.
float *forwardPropagation(float *input, float *y, float &cost){

	using namespace nous;

  for(int i = 0; i < layers; i++){
    calculateZ(input, i);
    calculateA(i, activation_functions[i]);
  }
  calculateC(y, cost, loss_function_index);

	return a[layers - 1];
}


//Back Propagation
void backwardPropagation(float *input, float *y){

	using namespace nous;

  calculateDLossToA(y, loss_function_index);

  for(int i = layers - 1; i >= 0; i--){
    calculateDAToZ(activation_functions[i], i);
    if(i > 0){
      calculateDZToA(i - 1);
    }

		if(batch_size != 1){
    	calculateDZToB(i);
    	calculateDZToW(input, i);
		}else{
			calculateDZToBStochastic(i);
      calculateDZToWStochastic(input, i);
		}
  }

	batch_counter++;

	for(int i = 0; i < layers; i++){
    memset(dz[i], 0, bzaRowsMapping[i] * sizeof(float));
    memset(da[i], 0, bzaRowsMapping[i] * sizeof(float));
  }
	
	if(batch_size != 1){
		if(batch_counter % batch_size == 0){

			//Calculate mean dw.
			for(int i = 0; i < layers; i++){
				for(int j = 0; j < weightsRowsMapping[i]; j++){
					for(int h = 0; h < weightsColumnsMapping[i]; h++){
						dw[i][j][h] /= batch_size;
					}
				}
			}

			//Calculate mean db.
			for(int i = 0; i < layers; i++){
				for(int j = 0; j < bzaRowsMapping[i]; j++){
					db[i][j] /= batch_size;
				}
			}

			//Update weights.
			updateW();

			//Update biases.
			updateB();

			//Set dw to 0.
			for(int i = 0; i < layers; i++){
   			for(int j = 0; j < weightsRowsMapping[i]; j++){
      		memset(dw[i][j], 0, weightsColumnsMapping[i] * sizeof(float));
    		}
  		}

			//Set db to 0.
  		for(int i = 0; i < layers; i++){
    		memset(db[i], 0, bzaRowsMapping[i] * sizeof(float));
  		}
		}
	}
}
