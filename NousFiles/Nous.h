#ifndef Nous
#define Nous

void initializeNN(int layers, int *neuronsMapping, int *activation_functions, int loss_function_index, float learning_rate, int batch_size);

void matrixMultiplication(float **twodArray, float *onedArray, float *resultArray, int m, int n);
void matrixSum(float *onedArray, float *twodArray2, float *resultArray, int m);
void matrixRelu(float *onedArray, float *resultArray, int m);

void initializeWeights(int *initialization_technique);
void initializeWeightsSaved(std::ifstream& inFile);
void initializeBiases();
void initializeBiasesSaved(std::ifstream& inFile);

float relu(float x);
float reluD(float x);
void softmax(float *onedArray, float *resultArray, int size);
float softMaxD(float *da, int i, int j);

void calculateZ(float *inputs, int z_layer);
void calculateA(int a_layer, int function_index);
void calculateC(float *y, float &cost, int loss_function_index);

void calculateDLossToA(float *y, int index_loss_function);
void calculateDAToZ(int index_activation_function, int index_layer);
void calculateDZToA(int index_layer);
void calculateDZToB(int index_layer);
void calculateDZToW(float *inputs, int index_layer);
void calculateDZToBStochastic(int index_layer);
void calculateDZToWStochastic(float *inputs, int index_layer);

void updateW();
void updateB();

float *forwardPropagation(float *input, float *y, float &cost);
void backwardPropagation(float *input, float *y);

#endif
