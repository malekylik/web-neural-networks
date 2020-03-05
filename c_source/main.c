#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <emscripten.h>

/*
#define ZLIB_WINAPI 
#define _SECURE_SCL_DEPRECATE 0

#include "lib\zlib\zlib.h"
*/

#define IMAGES_NUMBER_OFFEST 4
#define ROWS_NUMBER_OFFSET 8
#define COLUMNS_NUMBER_OFFSET 12
#define IMAGES_STRAR_PIXEL_OFFSET 16

#define LABLES_NUMBER_OFFEST 4
#define LABELS_VALUE_OFFSET 8

#define min(a, b) (a < b ? a : b)
#define to_negative_array_index(length, index) (length + index)

#define write_to_debug_file(str) fwrite(str, strlen(str), sizeof(char), fff);

// #define DEBUG_NT 0

typedef struct
{
	float* elements;
	int length;
} Vector;

typedef struct
{
	Vector** element;
	int row_length;
	int column_length;
} Matrix;

typedef struct
{
	int num_layers;
	int* sizes;
	Vector** biases;
	Matrix** weigths;
} Network;

typedef struct
{
	uint32_t image_number;
	uint32_t rows_number;
	uint32_t columns_number;
	float* images_buffer;
	uint8_t* labels_buffer;
} MNIST_data;

Vector* mat_mult(Matrix* m, Vector* v);
void add_two_vectors(Vector* v1, Vector* v2);
void sub_two_vectors(Vector* v1, Vector* v2);
void mult_two_vectors(Vector* v1, Vector* v2);
void div_two_vectors(Vector* v1, Vector* v2);
void add_two_matrices(Matrix* m1, Matrix* m2);
void sub_two_matrices(Matrix* m1, Matrix* m2);
void segmoid_vector(Vector* v);
uint32_t covert_bytes_to_int(uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4);
uint8_t get_label_for_image(MNIST_data* data, uint32_t image_number);
void update_mini_batch(Network* n, Vector** vs, Vector** outputs, uint32_t length, float eta);
Network backprop(Network* n, Vector* x, Vector* y);
void cost_derivative(Vector* output_activations, Vector* y);
Matrix* create_matrix(int column_length, int row_length);

#ifdef DEBUG_NT
FILE* fff = fopen("./log.txt", "w");
#endif

MNIST_data create_mnist_data(uint8_t* images_buffer, uint8_t* labels_buffer) {
	MNIST_data data;

	uint32_t image_number = covert_bytes_to_int(images_buffer[IMAGES_NUMBER_OFFEST + 0], images_buffer[IMAGES_NUMBER_OFFEST + 1], images_buffer[IMAGES_NUMBER_OFFEST + 2], images_buffer[IMAGES_NUMBER_OFFEST + 3]);
	uint32_t rows_number = covert_bytes_to_int(images_buffer[ROWS_NUMBER_OFFSET + 0], images_buffer[ROWS_NUMBER_OFFSET + 1], images_buffer[ROWS_NUMBER_OFFSET + 2], images_buffer[ROWS_NUMBER_OFFSET + 3]);
	uint32_t columns_number = covert_bytes_to_int(images_buffer[COLUMNS_NUMBER_OFFSET + 0], images_buffer[COLUMNS_NUMBER_OFFSET + 1], images_buffer[COLUMNS_NUMBER_OFFSET + 2], images_buffer[COLUMNS_NUMBER_OFFSET + 3]);

	uint32_t label_number = covert_bytes_to_int(labels_buffer[LABLES_NUMBER_OFFEST + 0], labels_buffer[LABLES_NUMBER_OFFEST + 1], labels_buffer[LABLES_NUMBER_OFFEST + 2], labels_buffer[LABLES_NUMBER_OFFEST + 3]);

	uint32_t data_number = min(min(image_number, label_number), 100);

	float* images_buffer_f = (float*)malloc(sizeof(float) * data_number * rows_number * columns_number);

	for (uint32_t i = 0; i < data_number * rows_number * columns_number; i++) {
		uint8_t temp = images_buffer[IMAGES_STRAR_PIXEL_OFFSET + i];
		float temp2 = temp / 256.0f;
		images_buffer_f[IMAGES_STRAR_PIXEL_OFFSET + i] = temp2;
	}

	data.rows_number = rows_number;
	data.columns_number = columns_number;
	data.image_number = data_number;
	data.images_buffer = images_buffer_f;
	data.labels_buffer = labels_buffer;

	return data;
}

uint8_t get_label_for_image(MNIST_data* data, uint32_t image_number) {
	assert(image_number < data->image_number);

	return data->labels_buffer[LABELS_VALUE_OFFSET + image_number];
}

float get_pixel_of_image(MNIST_data* data, uint32_t image_number, uint32_t pixel_offset) {
	assert(pixel_offset < data->rows_number * data->columns_number);

	return data->images_buffer[IMAGES_STRAR_PIXEL_OFFSET + (image_number * data->rows_number * data->columns_number) + pixel_offset];
}

Vector* create_vector_rand(int length) {
	Vector* v = (Vector*)malloc(sizeof(Vector));
	float* f = (float*)malloc(sizeof(float)* length);

	for (int i = 0; i < length; i++) {
		f[i] = ((float)rand() / (float)RAND_MAX) * 6 - 3;
	}

	v->length = length;
	v->elements = f;

	return v;
}

Vector* create_vector(int length) {
	Vector* v = (Vector*)malloc(sizeof(Vector));
	float* f = (float*)malloc(sizeof(float)* length);

	for (int i = 0; i < length; i++) {
		f[i] = 0;
	}

	v->length = length;
	v->elements = f;

	return v;
}

Vector* copy_vector(Vector* from) {
	Vector* v = (Vector*)malloc(sizeof(Vector));
	float* f = (float*)malloc(sizeof(float) * from->length);

	for (int i = 0; i < from->length; i++) {
		f[i] = from->elements[i];
	}

	v->length = from->length;
	v->elements = f;

	return v;
}

Matrix* copy_matrix(Matrix* from) {
	Matrix* m = (Matrix*)malloc(sizeof(Matrix));
	m->element = (Vector**)malloc(sizeof(Vector*)* from->column_length);
	m->row_length = from->row_length;
	m->column_length = from->column_length;

	for (int i = 0; i < from->column_length; i++) {
		m->element[i] = copy_vector(from->element[i]);
	}

	return m;
}

Vector** copy_biases(Network* n) {
	Vector** copy = (Vector**)malloc(sizeof(Vector*)* n->num_layers - 1);

	for (int32_t i = 0; i < n->num_layers - 1; i++) {
		// copy[i] = copy_vector(n->biases[i]);
		copy[i] = create_vector(n->biases[i]->length);
	}

	return copy;
}

Matrix** copy_weigths(Network* n) {
	Matrix** copy = (Matrix**)malloc(sizeof(Matrix*)* n->num_layers - 1);

	for (int32_t i = 0; i < n->num_layers - 1; i++) {
		//copy[i] = copy_matrix(n->weigths[i]);
		copy[i] = create_matrix(n->weigths[i]->column_length, n->weigths[i]->row_length);

		for (int32_t j = 0; j < n->weigths[i]->column_length; j++) {
			copy[i]->element[j] = create_vector(n->weigths[i]->row_length);
		}
	}

	return copy;
}

void free_vector(Vector* v) {
	free(v->elements);
	free(v);
}

void free_matrix(Matrix* m) {
	for (int32_t i = 0; i < m->column_length; i++) {
		free_vector(m->element[i]);
	}
	free(m->element);
	free(m);
}

Matrix* create_matrix_rand(int column_length, int row_length) {
	Matrix* m = (Matrix*)malloc(sizeof(Matrix));
	m->element = (Vector**)malloc(sizeof(Vector*)* column_length);
	m->row_length = row_length;
	m->column_length = column_length;

	for (int i = 0; i < column_length; i++) {
		m->element[i] = create_vector_rand(row_length);
	}

	return m;
}

Matrix* create_matrix(int column_length, int row_length) {
	Matrix* m = (Matrix*)malloc(sizeof(Matrix));
	m->element = (Vector**)malloc(sizeof(Vector*)* column_length);
	m->row_length = row_length;
	m->column_length = column_length;

	for (int i = 0; i < column_length; i++) {
		m->element[i] = 0;
		//m->element[i] = create_vector(row_length);
	}

	return m;
}

void show_vector(Vector* v) {
	for (int i = 0; i < v->length; i++) {
		printf("elem %i: %f\n", i, v->elements[i]);
	}
}

void show_biases(Network* network) {
	for (int i = 0; i < network->num_layers - 1; i++) {
		printf("vector %i\n", i);
		show_vector(network->biases[i]);
	}
}

void show_matrix(Matrix* m) {
	for (int i = 0; i < m->column_length; i++) {
		printf("row %i\n", i);
		show_vector(m->element[i]);
	}
}

uint32_t get_uncompressed_size(char file_path[]) {
	FILE* compressed_file = fopen(file_path, "rb");

	fseek(compressed_file, 0, SEEK_END);

	long compressed_size = ftell(compressed_file);

	fseek(compressed_file, -4, SEEK_CUR);

	uint8_t s[4] = { 0 };

	fread(&s[3], sizeof(int8_t), 1, compressed_file);
	fread(&s[2], sizeof(int8_t), 1, compressed_file);
	fread(&s[1], sizeof(int8_t), 1, compressed_file);
	fread(&s[0], sizeof(int8_t), 1, compressed_file);

	uint32_t uncompressed_size = (s[0] << 24) | (s[1] << 16) + (s[2] << 8) + s[3];

	fclose(compressed_file);

	return uncompressed_size;
}

uint8_t* read_file(FILE* file, uint32_t size) {
	fseek(file, 0, SEEK_SET);

	uint8_t* buff = (uint8_t*)malloc(sizeof(uint8_t) * size);

	fread(buff, sizeof(uint8_t), size, file);

	return buff;
}


uint32_t get_file_size(FILE* file) {
	fseek(file, 0, SEEK_END);

	return ftell(file);
}
 
Network* create_network(int* sizes, int length) {
	Network* n = (Network*)malloc(sizeof(Network));

	Vector** biases = (Vector**)malloc(sizeof(Vector*)* (length - 1));
	Matrix** weigths = (Matrix**)malloc(sizeof(Matrix*)* (length - 1));

	for (int i = 1; i < length; i++) {
		biases[i - 1] = create_vector_rand(sizes[i]);
		//biases[i - 1] = create_vector(sizes[i]);
	}

	for (int i = 1; i < length; i++) {
		weigths[i - 1] = create_matrix_rand(sizes[i], sizes[i - 1]);
		//weigths[i - 1] = create_matrix(sizes[i], sizes[i - 1]);

		//for (int j = 0; j < weigths[i - 1]->column_length; j++) {
		//	weigths[i - 1]->element[j] = create_vector(weigths[i - 1]->row_length);
		//}
	}

	n->num_layers = length;
	n->sizes = sizes;
	n->biases = biases;
	n->weigths = weigths;

	return n;
}

void set_bias(Network* n, float* b, int number, int length) {
	assert(number < n->num_layers - 1);

	free(n->biases[number]->elements);
	n->biases[number]->elements = b;
}

void set_matrix(Network* n, float** b, int number, int column_length) {
	assert(number < n->num_layers - 1);

	n->weigths[number]->column_length = column_length;

	for (int i = 0; i < n->weigths[number]->column_length; i++) {
		free(n->weigths[number]->element[i]->elements);
		n->weigths[number]->element[i]->elements = b[i];
	}
}

int get_network_num_layer(Network* n) {
	return n->num_layers;
}


int* get_network_sizes(Network* n) {
	return n->sizes;
}

Matrix* get_network_weight(Network* n, int number) {
	assert(number < n->num_layers - 1);

	return n->weigths[number];
}

Vector* get_column_from_matrix(Matrix* m, int number) {
	assert(number < m->column_length);

	return m->element[number];
}

void set_bias_value(Network* n, int bias_number, int elem,  float number) {
	assert(bias_number < n->num_layers - 1);
	assert(elem < n->biases[bias_number]->length);

	n->biases[bias_number]->elements[elem] = number;
}

float get_bias_value(Network* n, int bias_number, int elem) {
	assert(bias_number < n->num_layers - 1);
	assert(elem < n->biases[bias_number]->length);

	return n->biases[bias_number]->elements[elem];
}

void set_weigth_value(Network* n, int weigth_number, int row_number, int column_number, float number) {
	assert(weigth_number < n->num_layers - 1);
	assert(row_number < n->weigths[weigth_number]->column_length);
	assert(column_number < n->weigths[weigth_number]->element[row_number]->length);

	n->weigths[weigth_number]->element[row_number]->elements[column_number] = number;
}

float get_weigth_value(Network* n, int weigth_number, int row_number, int column_number) {
	assert(weigth_number < n->num_layers - 1);
	assert(row_number < n->weigths[weigth_number]->column_length);
	assert(column_number < n->weigths[weigth_number]->element[row_number]->length);

	return n->weigths[weigth_number]->element[row_number]->elements[column_number];
}

int get_columns_length_from_matrix(Matrix* m) {
	return m->column_length;
}

float* get_elements_from_vector(Vector* v) {
	return v->elements;
}

int get_elements_length_from_vector(Vector* v) {
	return v->length;
}

float sigmoid(float z) {
	return 1.0f / (1.0f + (float)exp(-z));
}

void sigmoid_prime(Vector* z) {
	segmoid_vector(z);

	Vector* prime = create_vector(z->length);

	for (int32_t i = 0; i < prime->length; i++) {
		prime->elements[i] = 1.0f;
	}

	sub_two_vectors(prime, z);
	mult_two_vectors(z, prime);

	free_vector(prime);
}
 
Vector* feedforward(Network* n, Vector* a) {
	Vector* temp = copy_vector(a);
	Vector* prev = NULL;

	for (int i = 0; i < n->num_layers - 1; i++) {
		prev = temp;
		temp = mat_mult(n->weigths[i], temp);
		add_two_vectors(temp, n->biases[i]);
		segmoid_vector(temp);

		free_vector(prev);
	}

	return temp;
}

void set_vector_value(Vector* v, int elem, float number) {
	assert(elem < v->length);

	v->elements[elem] = number;
}

int get_max_value_index_in_vector(Vector* v) {
	if (v->length < 1) return -1;

	int index = 0;
	float max = v->elements[0];

	for (int i = 1; i < v->length; i++) {
		if (max < v->elements[i]) {
			max = v->elements[i];
			index = i;
		}
	}

	return index;
}

Vector* mat_mult(Matrix* m, Vector* v) {
	assert(m->row_length == v->length);

	Vector* res = create_vector(m->column_length);

	for (int i = 0; i < m->column_length; i++) {
		for (int j = 0; j < v->length; j++) {
				res->elements[i] += m->element[i]->elements[j] * v->elements[j];
		}
	}

	return res;
}

Vector* mat_mult_transpose(Matrix* m, Vector* v) {
	assert(m->column_length == v->length);

	Vector* res = create_vector(m->row_length);

	for (int i = 0; i < m->row_length; i++) {
		for (int j = 0; j < v->length; j++) {
			res->elements[i] += m->element[j]->elements[i] * v->elements[j];
		}
	}

	return res;
}

void add_scal_to_vector(Vector* v, float value) {
	for (int i = 0; i < v->length; i++) {
		v->elements[i] += value;
	}
}

void mult_scal_to_vector(Vector* v, float value) {
	for (int i = 0; i < v->length; i++) {
		v->elements[i] *= value;
	}
}

void mult_scal_to_matrix(Matrix* m, float value) {
	for (int i = 0; i < m->column_length; i++) {
		for (int j = 0; j < m->row_length; j++) {
			m->element[i]->elements[j] *= value;
		}
	}
}

void add_two_vectors(Vector* v1, Vector* v2) {
	assert(v1->length == v2->length);

	for (int i = 0; i < v1->length; i++) {
		v1->elements[i] += v2->elements[i];
	}
}

void sub_two_vectors(Vector* v1, Vector* v2) {
	assert(v1->length == v2->length);

	for (int i = 0; i < v1->length; i++) {
		v1->elements[i] -= v2->elements[i];
	}
}

void mult_two_vectors(Vector* v1, Vector* v2) {
	assert(v1->length == v2->length);

	for (int i = 0; i < v1->length; i++) {
		v1->elements[i] *= v2->elements[i];
	}
}

void div_two_vectors(Vector* v1, Vector* v2) {
	assert(v1->length == v2->length);

	for (int i = 0; i < v1->length; i++) {
		v1->elements[i] /= v2->elements[i];
	}
}

void add_two_matrices(Matrix* m1, Matrix* m2) {
	assert(m1->column_length == m2->column_length && m1->row_length == m2->row_length);

	for (int i = 0; i < m1->column_length; i++) {
		for (int j = 0; j < m1->row_length; j++) {
			m1->element[i]->elements[j] += m2->element[i]->elements[j];
		}
	}
}

void sub_two_matrices(Matrix* m1, Matrix* m2) {
	assert(m1->column_length == m2->column_length && m1->row_length == m2->row_length);

	for (int i = 0; i < m1->column_length; i++) {
		for (int j = 0; j < m1->row_length; j++) {
			m1->element[i]->elements[j] -= m2->element[i]->elements[j];
		}
	}
}

uint32_t covert_bytes_to_int(uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4) {
	return (b1 << 24) | (b2 << 16) + (b3 << 8) + b4;
}

void segmoid_vector(Vector* v) {
	for (int i = 0; i < v->length; i++) {
		float temp = v->elements[i];
		float temp2 = sigmoid(temp);
		v->elements[i] = sigmoid(v->elements[i]);
	}
}

void swap(int *a, int *b) {
	int temp = *a;
	*a = *b;
	*b = temp;
}

void randomize(int arr[], int n) {
	srand(time(NULL));
	int i;
	for (i = n - 1; i > 0; i--) {
		int j = rand() % (i + 1);
		swap(&arr[i], &arr[j]);
	}
}

int compare_ints(const void* a, const void* b)
{
	int* arg1 = (int*)a;
	int* arg2 = (int*)b;

	return *arg1 - *arg2;
}

void shuffle_vector_array(Vector** array, Vector** array1, size_t count) {
	uint32_t* numbers = (uint32_t*)malloc(sizeof(uint32_t) * count);

	for (uint32_t i = 0; i < count; i ++) {
		numbers[i] = i;
	}

	randomize((int*)numbers, count);

	Vector* temp = NULL;

	for (uint32_t i = 0; i < count; i++) {
		temp = array[numbers[i]];
		array[numbers[i]] = array[i];
		array[i] = temp;

		temp = array1[numbers[i]];
		array1[numbers[i]] = array1[i];
		array1[i] = temp;
	}

	free(numbers);
}

char* get_vector_as_string(char vector_name[], Vector* v) {
#define DIGITS 20

	uint32_t name_length = strlen(vector_name);
	uint32_t str_length = name_length + v->length * DIGITS;
	char* str = (char*)malloc(sizeof(char)* str_length);
	uint32_t i = 0;

	for (uint32_t j = 0; j < str_length; j++) {
		str[j] = ' ';
	}

	for (; i < name_length; i++) {
		str[i] = vector_name[i];
	}

	uint32_t element = 0;

	for (; i < str_length && element < v->length; i++) {
		int line = element % 1000;
		float val = v->elements[element];

		int len = sprintf(str + i, "%i: %f\n", element, val);
		str[i + len] = ' ';
		i += len - 1;
		element += 1;
	}

	str[str_length - 1] = '\0';

	return str;
}

void SGD(Network* n, MNIST_data* training_data, uint32_t epochs, uint32_t mini_batch_size, float eta) {
	int gg = 0;

	Vector** vs = (Vector**)malloc(sizeof(Vector*) * training_data->image_number);

	for (uint32_t i = 0; i < training_data->image_number; i++) {
		uint32_t length = training_data->rows_number * training_data->columns_number;
		vs[i] = create_vector(length);

		for (uint32_t j = 0; j < length; j++) {
			vs[i]->elements[j] = (float)get_pixel_of_image(training_data, i, j);
		}
	}

	Vector** outputs = (Vector**)malloc(sizeof(Vector*)* training_data->image_number);

	for (uint32_t i = 0; i < training_data->image_number; i++) {
		uint32_t length = 10;
		outputs[i] = create_vector(length);

		outputs[i]->elements[get_label_for_image(training_data, i)] = 1.0f;
	}

#ifdef FALSE
	char d[30];
	sprintf(d, "--%files\n--");

	write_to_debug_file(d);

	for (uint32_t biases = 0; biases < training_data->image_number; biases++) {
		sprintf(d, "%i file\n", biases);

		write_to_debug_file(d);
		char* str = get_vector_as_string(d, vs[biases]);

		write_to_debug_file(str);

		free(str);

		sprintf(d, "\n%i output\n", biases);
		str = get_vector_as_string(d, outputs[biases]);
		
		write_to_debug_file(str);

		free(str);
	}

	write_to_debug_file("\n\n\n");

#endif

	for (uint32_t j = 0; j < epochs; j++) {
		time_t rawtime;
		struct tm * timeinfo;
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		
		//shuffle_vector_array(vs, outputs, training_data->image_number);

		uint32_t mini_batches_number = training_data->image_number / mini_batch_size;

		for (uint32_t i = 0; i < mini_batches_number; i++) {
			
			size_t offset = (i * mini_batch_size);

			update_mini_batch(n, vs + offset, outputs + offset, min(mini_batch_size, training_data->image_number - mini_batch_size * i), eta);

#ifdef DEBUG_NT
			char d[30];
			sprintf(d, "--%i batch\n--", i);

			write_to_debug_file(d);

			for (uint32_t biases = 0; biases < n->num_layers - 1; biases++) {
				sprintf(d, "%i biases\n", biases);

				write_to_debug_file(d);
				char* str = get_vector_as_string(d, n->biases[biases]);

				write_to_debug_file(str);

				free(str);
			}

			write_to_debug_file("\n\n\n");

#endif
		}

		printf("epoch complete %i / %i\n", j, epochs);
		printf("start in %s\n", asctime(timeinfo));
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		printf("end in %s\n", asctime(timeinfo));
	}

	for (uint32_t i = 0; i < training_data->image_number; i++) {
		free_vector(vs[i]);
		free_vector(outputs[i]);
	}

	free(vs);
	free(outputs);
}

void update_mini_batch(Network* n, Vector** vs, Vector** outputs, uint32_t length, float eta) {
	Vector** nabla_b = copy_biases(n);
	Matrix** nabla_w = copy_weigths(n);

	float etaDivLength = eta / length;

	for (uint32_t i = 0; i < length; i++) {
		Vector* x = vs[i];
		Vector* y = outputs[i];

		Network new_network = backprop(n, x, y);

		Vector** delta_nabla_b = new_network.biases;
		Matrix** delta_nabla_w = new_network.weigths;

		//printf("update_mini_batch %i\n", i);
		//for (int32_t j = 0; j < new_network.num_layers - 1; j++) {
		//	printf("delta_nabla_b %i\n", j);
		//	show_vector(delta_nabla_b[j]);
	//	}

		for (int32_t j = 0; j < new_network.num_layers - 1; j++) {
			add_two_vectors(nabla_b[j], delta_nabla_b[j]);
		}

		for (int32_t j = 0; j < new_network.num_layers - 1; j++) {
			add_two_matrices(nabla_w[j], delta_nabla_w[j]);
		}

		// free
		for (int32_t j = 0; j < new_network.num_layers - 1; j++) {
			free_vector(delta_nabla_b[j]);
		}
	free(delta_nabla_b);
//
		for (int32_t j = 0; j < new_network.num_layers - 1; j++) {
			free_matrix(delta_nabla_w[j]);
		}
	free(delta_nabla_w);

		//printf("biases on i: %i\n", i);
		
		//for (int32_t i = 0; i < new_network.num_layers - 1; i++) {
		//	show_vector(nabla_b[i]);
		//}
	}

	for (int32_t i = 0; i < n->num_layers - 1; i++) {
		mult_scal_to_vector(nabla_b[i], etaDivLength);
		sub_two_vectors(n->biases[i], nabla_b[i]);
		//show_vector(nabla_b[i]);
	}


	for (int32_t i = 0; i < n->num_layers - 1; i++) {
		mult_scal_to_matrix(nabla_w[i], etaDivLength);
		sub_two_matrices(n->weigths[i], nabla_w[i]);
	}

	
	//for (uint16_t i = 0; i < n->num_layers - 1; i++) {
	//show_vector(nabla_b[i]);
	//}

	for (int32_t i = 0; i < n->num_layers - 1; i++) {
		free_vector(nabla_b[i]);
	}
	free(nabla_b);

	for (int32_t i = 0; i < n->num_layers - 1; i++) {
		free_matrix(nabla_w[i]);
	}
	free(nabla_w);

	//printf("butch size: %i\n", length);
}

// x - 784 px, y - [1..10] - network predict of number
Network backprop(Network* n, Vector* x, Vector* y) {
	Vector* activation = copy_vector(x);
	Vector** activations = (Vector**)malloc(sizeof(Vector*)* n->num_layers);
	Vector** zs = (Vector**)malloc(sizeof(Vector*)* n->num_layers - 1);
	Vector** nabla_b = (Vector**)malloc(sizeof(Vector*)* n->num_layers - 1);
	Matrix** nabla_w = (Matrix**)malloc(sizeof(Matrix*)* n->num_layers - 1);

	uint32_t num_layers = n->num_layers;
	uint32_t num_biases = num_layers - 1;

	activations[0] = activation;

	for (int32_t i = 0; i < n->num_layers - 1; i++) {
		activation = mat_mult(n->weigths[i], activation);

		add_two_vectors(activation, n->biases[i]);

		zs[i] = copy_vector(activation);

		segmoid_vector(activation);

		activations[i + 1] = activation;
	}

	Vector* zSub1 = copy_vector(zs[to_negative_array_index(num_biases, -1)]);
	sigmoid_prime(zSub1);
	Vector* delta = copy_vector(activations[to_negative_array_index(num_layers, -1)]);
	cost_derivative(delta, y);

	mult_two_vectors(delta, zSub1);
	free_vector(zSub1);

	nabla_b[to_negative_array_index(num_biases, -1)] = delta;
	Vector* delta_w = copy_vector(delta);

	Matrix* m = create_matrix(delta_w->length, activations[to_negative_array_index(num_layers, -2)]->length);

	for (int32_t i = 0; i < delta_w->length; i++) {
		m->element[i] = copy_vector(activations[to_negative_array_index(num_layers, -2)]);
		mult_scal_to_vector(m->element[i], delta_w->elements[i]);
	}

	nabla_w[to_negative_array_index(num_biases, -1)] = m;
	free_vector(delta_w);

	for (int32_t i = 2; i < n->num_layers; i++) {
		zSub1 = copy_vector(zs[to_negative_array_index(num_biases, -i)]);
		sigmoid_prime(zSub1);

		delta = mat_mult_transpose(n->weigths[to_negative_array_index(num_biases, -i + 1)], delta);
		
		mult_two_vectors(delta, zSub1);

		nabla_b[to_negative_array_index(num_biases, -i)] = delta;

		Vector* delta_w = copy_vector(delta);

		Matrix* m = create_matrix(delta_w->length, activations[to_negative_array_index(num_layers, -i - 1)]->length);

		for (int32_t j = 0; j < delta_w->length; j++) {
			m->element[j] = copy_vector(activations[to_negative_array_index(num_layers, -i - 1)]);
			mult_scal_to_vector(m->element[j], delta_w->elements[j]);
		}

		nabla_w[to_negative_array_index(num_biases, -i)] = m;

		free_vector(zSub1);
		free_vector(delta_w);
	}

	for (uint32_t i = 0; i < num_layers; i++) {
		free_vector(activations[i]);
	}
	free(activations);

	for (uint32_t i = 0; i < num_biases; i++) {
		free_vector(zs[i]);
	}
	free(zs);

	Network nn = { num_layers, n->sizes, nabla_b, nabla_w };

	return nn;
}

void cost_derivative(Vector* output_activations, Vector* y) {
	assert(output_activations->length == y->length);

	for (int32_t i = 0; i < output_activations->length; i++) {
		output_activations->elements[i] -= y->elements[i];
	}
}

// int main() {
// 	srand((unsigned int)time(NULL));

// 	FILE* image_file = fopen("./data/train-images.idx3-ubyte", "rb");
// 	uint32_t image_file_size = get_file_size(image_file);
// 	uint8_t* image_buff = read_file(image_file, image_file_size);
// 	fclose(image_file);

// 	FILE* label_file = fopen("./data/train-labels.idx1-ubyte", "rb");
// 	uint32_t label_size = get_file_size(label_file);
// 	uint8_t* label_buff = read_file(label_file, label_size);
// 	fclose(label_file);

// 	MNIST_data data = create_mnist_data(image_buff, label_buff);

// 	free(image_buff);

// 	uint32_t image_number = 2;

// 	Vector check;

// 	check.length = 784;
// 	check.elements = (float*)malloc(sizeof(float)* 784);

// 	for (uint32_t i = 0; i < 784; i++) {
// 		check.elements[i] = get_pixel_of_image(&data, image_number, i);
// 	}

// 	//char* vec = get_vector_as_string("check", &check);

// 	//write_to_debug_file(vec);

// 	//uint8_t f = get_label_for_image(&data, 0);
// 	//uint8_t f1 = get_label_for_image(&data, 1);

// 	int sizes[] = { 784, 30, 10 };

// 	Network* network = create_network(sizes, 3);
// 	Vector* v = create_vector_rand(2);

// 	SGD(network, &data, 5, 10, 3.0f);

// #ifdef DEBUG_NT
// 	fclose(fff);
// #endif

// 	show_biases(network);

// 	show_vector(feedforward(network, &check));
// 	printf("answer %i", get_label_for_image(&data, image_number));

// 	_getch();

// 	return 0;
// }
