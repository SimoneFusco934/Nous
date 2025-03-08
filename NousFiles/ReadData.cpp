#include <iostream>
#include <fstream>

int reverseInt(int i){
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void initialize_read_image(ifstream &file, int &magic_number, int &number_of_images, int &n_rows, int &n_cols, int index){
	if(index == 0){
  	file.open("train-images.idx3-ubyte");
	}else{
		file.open("t10k-images.idx3-ubyte");
	}
  if(file){
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
  }
}

void initialize_read_label(ifstream &file, int &magic_number, int &number_of_labels, int index){
	if(index == 0){
  	file.open("train-labels.idx1-ubyte");
	}else{
		file.open("t10k-labels.idx1-ubyte");
	}
  if(file){
    int magic_number=0;
    int number_of_labels=0;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_labels,sizeof(number_of_labels));
    number_of_labels = reverseInt(number_of_labels);
  }
}

void read_image(ifstream &file, int n_rows, int n_cols, float *input){
  for(int r=0;r<n_rows;r++){
    for(int c=0;c<n_cols;c++){
      unsigned char temp = 0;
      file.read((char*)&temp,sizeof(temp));
      //float num = (temp / 127.5) - 1;
			//values are between [0, 1]
			float num = temp / 255.5 ;
    	input[r*28 + c] = num;
		}
	}
}

void read_label(ifstream &file, int &lab){
  unsigned char temp = 0;
  file.read((char*)&temp,sizeof(temp));
	lab = temp;
}
