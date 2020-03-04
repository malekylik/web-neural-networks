var IMAGES_NUMBER_OFFEST = 4;
var ROWS_NUMBER_OFFSET = 8;
var COLUMNS_NUMBER_OFFSET = 12;
var IMAGES_STRAR_PIXEL_OFFSET = 16;

var LABLES_NUMBER_OFFEST = 4;
var LABELS_VALUE_OFFSET = 8;

function min(a, b) {
    return a < b ? a : b;
}

function covert_bytes_to_int(b1, b2, b3, b4) {
	return (b1 << 24) | (b2 << 16) + (b3 << 8) + b4;
}

function create_mnist_data(images_buffer, labels_buffer) {
    var data = {};

    var image_number = covert_bytes_to_int(images_buffer[IMAGES_NUMBER_OFFEST + 0], images_buffer[IMAGES_NUMBER_OFFEST + 1], images_buffer[IMAGES_NUMBER_OFFEST + 2], images_buffer[IMAGES_NUMBER_OFFEST + 3]);
	var rows_number = covert_bytes_to_int(images_buffer[ROWS_NUMBER_OFFSET + 0], images_buffer[ROWS_NUMBER_OFFSET + 1], images_buffer[ROWS_NUMBER_OFFSET + 2], images_buffer[ROWS_NUMBER_OFFSET + 3]);
	var columns_number = covert_bytes_to_int(images_buffer[COLUMNS_NUMBER_OFFSET + 0], images_buffer[COLUMNS_NUMBER_OFFSET + 1], images_buffer[COLUMNS_NUMBER_OFFSET + 2], images_buffer[COLUMNS_NUMBER_OFFSET + 3]);

	var label_number = covert_bytes_to_int(labels_buffer[LABLES_NUMBER_OFFEST + 0], labels_buffer[LABLES_NUMBER_OFFEST + 1], labels_buffer[LABLES_NUMBER_OFFEST + 2], labels_buffer[LABLES_NUMBER_OFFEST + 3]);

    var data_number = min(100, min(image_number, label_number));
    
    var images_buffer_f = new Float32Array(data_number * rows_number * columns_number);

	for (var i = 0; i < data_number * rows_number * columns_number; i++) {
		var temp = images_buffer[IMAGES_STRAR_PIXEL_OFFSET + i];
		var temp2 = temp / 256.0;
		images_buffer_f[IMAGES_STRAR_PIXEL_OFFSET + i] = temp2;
	}

	data.rows_number = rows_number;
	data.columns_number = columns_number;
	data.image_number = data_number;
	data.images_buffer = images_buffer_f;
	data.labels_buffer = labels_buffer;

	return data;
}

function get_label_for_image(data, image_number) {
	return data.labels_buffer[LABELS_VALUE_OFFSET + image_number];
}

function get_pixel_of_image(data, image_number, pixel_offset) {
	return data.images_buffer[IMAGES_STRAR_PIXEL_OFFSET + (image_number * data.rows_number * data.columns_number) + pixel_offset];
}

function get_pixel_of_image_by_cord(data, image_number, x, y) {
	return data.images_buffer[IMAGES_STRAR_PIXEL_OFFSET + (image_number * data.rows_number * data.columns_number) + (data.columns_number * y) + x];
}

function putPixel(buffer, x, y, value) {
    const color = value * 255;
  
    let offset = 4 * x + buffer.width * 4 * y;
    buffer.data[offset++] = color;
    buffer.data[offset++] = color;
    buffer.data[offset++] = color;
    buffer.data[offset++] = 255;
}


var scale_image = 20;
var image_data  = new ImageData(24 * scale_image, 24 * scale_image);
var input = document.getElementsByClassName('image_number_input')[0];

function update_image_data(mnist_data, image_number) {
    for (let i = 0; i < 24 * scale_image; i++) {
        for (let j = 0; j < 24 * scale_image; j++) {
            var temp = get_pixel_of_image_by_cord(mnist_data, image_number, 24 - ((j / scale_image) | 0), ((i / scale_image) | 0));
            putPixel(image_data, i, j, temp);
        }
    }
}

Module['onRuntimeInitialized'] = async function () {
    var files = await Promise.all([
        fetch('../network-configs/config.json'),
        fetch('../data/train-images.idx3-ubyte'),
        fetch('../data/train-labels.idx1-ubyte'),
    ]);

    var network_config = await files[0].json();
    var images = new Uint8Array(await files[1].arrayBuffer());
    var labels = new Uint8Array(await files[2].arrayBuffer());
    // Network* n, float** b, int number, int column_length
    var set_bias_value = Module.cwrap('set_bias_value', 'void', ['number', 'number', 'number', 'number']);
    var get_bias_value = Module.cwrap('get_bias_value', 'number', ['number', 'number', 'number']);
    var set_weigth_value = Module.cwrap('set_weigth_value', 'void', ['number', 'number', 'number', 'number', 'number']);
    var get_weigth_value = Module.cwrap('get_weigth_value', 'number', ['number', 'number', 'number', 'number']);
    var malloc = Module.cwrap('malloc', 'number', ['number']);

    console.log('config', network_config);

    var int_type_size = getNativeTypeSize('i32');

    var mnist_data = create_mnist_data(images, labels);

    var sizes_pointer = malloc(int_type_size * network_config.sizes.length);

    network_config.sizes.forEach((size, i) => setValue(sizes_pointer + (int_type_size * i), size, 'i32'));

    var network_pointer = Module.ccall('create_network', 'number', ['number', 'number'], [sizes_pointer, 3]);

    for (let i = 0; i < network_config.biases.length; i++) {
        for (let j = 0; j < network_config.biases[i].length; j++) {
            set_bias_value(network_pointer, i, j, network_config.biases[i][j]);
        }
    }


    for (let i = 0; i < network_config.weigths.length; i++) {
        for (let j = 0; j < network_config.weigths[i].length; j++) {
            for (let w = 0; w < network_config.weigths[i][j].length; w++) {
                set_weigth_value(network_pointer, i, j, w, network_config.weigths[i][j][w]);
            }
        }
    }

    input.addEventListener('change', () => {
        var image_number = Number(input.value);

        update_image_data(mnist_data, image_number);
        ctx.putImageData(image_data, 0, 0);
        console.log('answer', get_label_for_image(mnist_data, image_number));
    });

    console.log('answer', get_label_for_image(mnist_data, 0));

    var ctx = digit_canvas.getContext('2d');
    update_image_data(mnist_data, 0);
    ctx.putImageData(image_data, 0, 0);
}