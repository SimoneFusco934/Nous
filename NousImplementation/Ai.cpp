#include <iostream>
#include <fstream>
#include "Ai.h"
#include "ReadData.cpp"
#include "Nous.h"
#include "Nous.cpp"

using namespace std;

void testOnCanvas();
void saveToFile(int layers, int *neuronsMapping, int *activation_functions, int loss_function_index, float learning_rate, int batch_size);

const int input_layer_neurons = 784;
const int output_layer_neurons = 10;

ifstream train_image_file;
ifstream train_label_file;
ifstream test_image_file;
ifstream test_label_file;

int main(int argc, char *argv[]){

	//Initializing reading from data
  int magic_number_i, magic_number_l, number_of_images, number_of_labels, number_of_rows, number_of_columns, magic_number_i_t, magic_number_l_t, number_of_images_t, number_of_rows_t, number_of_columns_t, number_of_labels_t;
  initialize_read_image(train_image_file, magic_number_i, number_of_images, number_of_rows, number_of_columns, 0);
  initialize_read_label(train_label_file, magic_number_l, number_of_labels, 0);
  initialize_read_image(test_image_file, magic_number_i_t, number_of_images_t, number_of_rows_t, number_of_columns_t, 1);
  initialize_read_label(test_label_file, magic_number_l_t, number_of_labels_t, 1);

	int layers;
	int n_neurons[layers + 1];
	int activation_functions[layers];
	int loss_function_index;
	float learning_rate;
	int initialize_weights_technique[layers];
	int batch_s;
	int ep;

	//layers = 4, n_neurons = {784, 140, 140, 140, 10}, activation_functions = {0, 0, 0, 1}, loss_function_index = 1, learning_rate = 0.01, initialize_weights_technique = {0, 0, 0, 1}, batch_s = 20, ep = 20;

	cout << "How many layers? (without counting input layer) ";
	cin >> layers;
	cout << endl;

	for(int i = 0; i <= layers; i++){
		cout << "Insert number of neurons in layer " << i << ": ";
    cin >> n_neurons[i];
    cout << endl;
	}
	for(int i = 0; i < layers; i++){
		cout << "Insert activation function in layer " << i + 1 << " (0 = ReLU, 1 = Softmax): ";
		cin >> activation_functions[i];
		cout << endl;
	}

	for(int i = 0; i < layers; i++){
		cout << "Insert weights initialization technique in layer " << i + 1 << " (0 = He Normal, 1 = Xavier Normal): ";
    cin >> initialize_weights_technique[i];
    cout << endl;
  }

	cout << "Insert loss function (0 = Sum squared error, 1 = Cross entropy loss): ";
  cin >> loss_function_index;
  cout << endl;
  cout << "Insert learning rate: ";
  cin >> learning_rate;
  cout << endl;
	cout << "Insert batch size: ";
	cin >> batch_s;
	cout << endl;
	cout << "Insert number of epochs: ";
	cin >> ep;
	cout << endl;

	initializeNN(layers, n_neurons, activation_functions, loss_function_index, learning_rate, batch_s);

	//Initializing weights
	initializeWeights(initialize_weights_technique);

	//Initializing Biases
	initializeBiases();

	//Training
	//l = epochs

	for(int l = 0; l < ep; l++){
		train(number_of_rows, number_of_columns);

		cout << endl << "Training section: " << l << " completed." << endl << endl;

		//train_image_file.seekg(0, ios::beg);
		//train_label_file.seekg(0, ios::beg);
		train_image_file.close();
		train_label_file.close();

		int magic_number_i, magic_number_l, number_of_images, number_of_labels, number_of_rows, number_of_columns, magic_number_i_t, magic_number_l_t, number_of_images_t, number_of_rows_t, number_of_columns_t, number_of_labels_t;
  	initialize_read_image(train_image_file, magic_number_i, number_of_images, number_of_rows, number_of_columns, 0);
  	initialize_read_label(train_label_file, magic_number_l, number_of_labels, 0);
	}

	//Testing
	test(number_of_rows_t, number_of_columns_t);

	//testOnCanvas();
	saveToFile(layers, n_neurons, activation_functions, loss_function_index, learning_rate, batch_s);

	return 0;
}

void train(int number_of_rows, int number_of_columns){

	//Declaring y array
  float *y = new float[output_layer_neurons];

  //Declaring cost
  float cost;

  //Declaring input array and label
  float *input = new float[input_layer_neurons];
  int label;

	int counter = 0;

	for(int i = 0; i < 60000; i++){

    read_image(train_image_file, number_of_rows, number_of_columns, input);
    read_label(train_label_file, label);

    for(int j = 0; j < output_layer_neurons; j++){
      if(j == label){
        y[j] = 1;
      }else{
        y[j] = 0;
      }
    }

		float *res;
		res = forwardPropagation(input, y, cost);
		
		backwardPropagation(input, y);

		
		float max = 0;
    float index = 0;
		
    for(int p = 0; p < output_layer_neurons; p++){
      if(res[p] > max){
        max = res[p];
        index = p;
      }
    }

    if(label == index){
      counter++;
    }
		

    if(i % 1000 == 0 && i != 0){
      cout << "Epoch " << i / 1000 << ". Accuracy: " << (counter * 100) / 1000.0 << "%" << endl;
			counter = 0;
    }
  }
}

void test(int number_of_rows, int number_of_columns){

	//Declaring y array
  float *y = new float[output_layer_neurons];

  //Declaring cost
  float cost;

  //Declaring input array and label
  float *input = new float[input_layer_neurons];
  int label;

	int counter = 0;

	for(int l = 0; l < 10000; l++){

    read_image(test_image_file, number_of_rows, number_of_columns, input);
    read_label(test_label_file, label);

    for(int j = 0; j < output_layer_neurons; j++){
      if(j == label){
        y[j] = 1;
      }else{
        y[j] = 0;
      }
    }

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

    if(label == index){
      counter++;
    }else{
			for(int l = 0; l < number_of_rows; l++){
        for(int k = 0; k < number_of_columns; k++){
          if(abs(int((input[l*number_of_rows + k] * 255) / 127.5 - 1)) == 0){
            cout << "#" << " ";
          }else{
            cout << "  ";
          }
        }
        cout << endl;
      }
      cout << endl;
			cout << "Prediction: " << index << endl;
      cout << "Correct Answer: " << label << endl << endl;
		}


	}

	cout << endl << "Percentage of correct answers: " << counter * 100.0 / 10000 << "%" << endl;
}

void testOnCanvas(){

	//Declaring y array
  float *y = new float[output_layer_neurons];

  //Declaring cost
  float cost;			

	float *input = new float[input_layer_neurons];

	while(1){
	cout << endl << endl << "Insert number: ";

	for(int i = 0; i < 784; i++){
		cin >> input[i];
	}	

	cout << endl << "You entered: " << endl << endl;

	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			cout << input[i * 28 + j];
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

	cout << "Prediction: " << index << endl;
	}
}

void saveToFile(int layers, int *neuronsMapping, int *activation_functions, int loss_function_index, float learning_rate, int batch_size){

	using namespace nous;

	bool wantToSave = false;
	char answer;

	cout << "Do you want to save the result? y/n: ";
	cin >> answer;

	if(answer == 'y' || answer == 'Y'){
		wantToSave = true;
	}
		

	if(wantToSave){
  	std::ofstream outFile("saved.txt");  // "example.txt" is the file you want to write to

		// Check if the file is open and ready for writing
  	if (outFile.is_open()) {

  		outFile << layers << std::endl;  
			for(int i = 0; i < layers + 1; i++){
				outFile << neuronsMapping[i] << endl;
			}

			for(int i = 0; i < layers; i++){
        outFile << activation_functions[i] << endl;
      }

			outFile << loss_function_index << endl;
			outFile << learning_rate << endl;
			outFile << batch_size << endl;

			for(int i = 0; i < layers - 1; i++){

			}

			for(int i = 0; i < layers; i++){
      	for(int j = 0; j < weightsRowsMapping[i]; j++){
      		for(int h = 0; h < weightsColumnsMapping[i]; h++){
		      	outFile <<  weights[i][j][h] << endl;
    		  }
		    }
			}

			for(int i = 0; i < layers; i++){
    		for(int j = 0; j < bzaRowsMapping[i]; j++){
      		outFile << biases[i][j] << endl;
    		}
  		}


    	// Close the file after writing
	    outFile.close();
  	  std::cout << "Data written to the file successfully." << std::endl;
  	}else {
  		std::cout << "Failed to open the file!" << std::endl;  // Error handling if the file can't be opened
  	}
	}
}
























