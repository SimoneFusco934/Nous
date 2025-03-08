#ifndef Ai
#define Ai

using namespace std;

int reverseInt(int i);
void initialize_read_image(ifstream &file, int &magic_number, int &number_of_images, int &n_rows, int &n_cols, int index);
void read_image(ifstream &file, int n_rows, int n_cols, float *input);
void initialize_read_label(ifstream &file, int &magic_number, int &number_of_labels, int index);
void read_label(ifstream &file, int &lab);

//void train(float *input, float ***weights, float **biases, float **z, float **a, float **dz, float **da, float *y, float *c, float &cost, int layers, int *weightsRowsMapping, int *weightsColumnsMapping, int *bzaRowsMapping, int number_of_rows, int number_of_columns, int &label, int *activation_functions);
void train(int number_of_rows, int number_of_columns);
void test(int number_of_rows, int number_of_columns);

#endif
