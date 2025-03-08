#include <iostream>
#include <fstream>
#include <string>
#include "Ai.h"
#include "ReadData.cpp"
#include "Nous.h"
#include "Nous.cpp"

const int input_layer_neurons = 784;
const int output_layer_neurons = 10;


void testOnCanvas();

int main(){

	int layers;
  int loss_function_index;
  float learning_rate;
  int batch_s;

	// Open the file in input mode
  std::ifstream inFile("saved.txt");

  // Check if the file was opened successfully
  if(!inFile) {
  	std::cerr << "Error opening file!" << std::endl;
    return 1;  // Return an error code if the file cannot be opened
  }

  //std::string line;
    
	inFile >> layers;

	int n_neurons[layers + 1];
  int activation_functions[layers];
	int initialize_weights_technique[layers];

	for(int i = 0; i < layers + 1; i++){
		inFile >> n_neurons[i];
	}

	for(int i = 0; i < layers; i++){
    inFile >> activation_functions[i];
  }


	inFile >> loss_function_index;
	inFile >> learning_rate;
	inFile >> batch_s;

	//

	//

	initializeNN(layers, n_neurons, activation_functions, loss_function_index, learning_rate, batch_s);

	//Initializing weights
  initializeWeightsSaved(inFile);

  //Initializing Biases
  initializeBiasesSaved(inFile);

	// Close the file after reading
  inFile.close();

	testOnCanvas();
}

void testOnCanvas(){

  //Declaring y array
  float *y = new float[output_layer_neurons];

  //Declaring cost
  float cost;

  float *input = new float[input_layer_neurons];

	//bool stop = false;

  //while(!stop){
  cout << endl << endl << "Insert number: ";

  for(int i = 0; i < 784; i++){
    cin >> input[i];
  }

  cout << endl << "You entered: " << endl << endl;

  for(int i = 0; i < 28; i++){
    for(int j = 0; j < 28; j++){
			if(input[i * 28 + j] > 0.5){
				cout << "#";
			}else{
				cout << " ";
			}
      //cout << input[i * 28 + j];
    }

    cout << endl;
  }

  cout << endl << endl << endl;

  float *res;
  res = forwardPropagation(input, y, cost);

  float max = 0;
  float index = 0;

  for(int p = 0; p < output_layer_neurons; p++){
    if(res[p] > max){
      max = res[p];
      index = p;
    }
  }

  cout << "AI prediction: " << index << endl << endl;
/*
	cout << "Do you want to continue? [y/n]" << endl;

	char ans;

	cin >> ans;

	if(ans == 'n' || ans == 'N'){
		stop = true;
	}
 }*/
}
