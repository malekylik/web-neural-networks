console.log(Module);

Module['onRuntimeInitialized'] = async function () {
    var network_config = await (await fetch('../network-configs/config.json')).json();
    // Network* n, float* b, int number, int length
    var set_bias = Module.cwrap('set_bias', 'void', ['number', 'number', 'number', 'number']);
    // Network* n, float** b, int number, int column_length
    var set_matrix = Module.cwrap('set_bias', 'void', ['number', 'number', 'number', 'number']);
    var get_network_num_layer = Module.cwrap('get_network_num_layer', 'number', ['number']);
    var get_network_weight = Module.cwrap('get_network_weight', 'number', ['number', 'number']);
    var get_column_from_matrix = Module.cwrap('get_column_from_matrix', 'number', ['number', 'number']);
    var get_elements_from_vector = Module.cwrap('get_elements_from_vector', 'number', ['number', 'number']);
    var get_columns_length_from_matrix = Module.cwrap('get_columns_length_from_matrix', 'number', ['number']);
    var get_elements_length_from_vector = Module.cwrap('get_elements_length_from_vector', 'number', ['number']);

    console.log('config', network_config);

    var malloc = Module.cwrap('malloc', 'number', ['number']);
    var int_type_size = getNativeTypeSize('i32');
    var float_type_size = getNativeTypeSize('float');

    var sizes_pointer = malloc(int_type_size * network_config.sizes.length);

    network_config.sizes.forEach((size, i) => setValue(sizes_pointer + (int_type_size * i), size, 'i32'));

    var network_pointer = Module.ccall('create_network', 'number', ['number', 'number'], [sizes_pointer, 3]);
    var _sizes = Module.ccall('get_network_sizes', 'number', ['number'], [network_pointer]);

    console.log(network_config.biases.length);

    for (let i = 0; i < network_config.biases.length; i++) {
        var bias_pointer = malloc(float_type_size * network_config.biases[i].length);

        for (let j = 0; j < network_config.biases[i].length; j++) {
            setValue(bias_pointer + (float_type_size * j), network_config.biases[i][j], 'float');
        }

        set_bias(network_pointer, bias_pointer, i, network_config.biases[i].length);
    }

    for (let i = 0; i < network_config.weigths.length; i++) {
        var columns_pointer = malloc(int_type_size * network_config.weigths[i].length);
        for (let j = 0; j < network_config.weigths[i].length; j++) {
            var column_pointer = malloc(float_type_size * network_config.weigths[i][j].length);

            setValue(columns_pointer + (j * int_type_size), column_pointer, 'i32');

            for (let w = 0; w < network_config.weigths[i][j].length; w++) {
                setValue(column_pointer + (j * int_type_size) + (float_type_size * w), network_config.weigths[i][j][w], 'float');
            }
        }

        set_matrix(network_pointer, columns_pointer, i, network_config.weigths[i].length);
    }

    var num_layer = get_network_num_layer(network_pointer);

    if (network_config.weigths.length !== num_layer - 1) {
        throw new Error('!weigths.length');
    }


    for (let i = 0; i < num_layer - 1; i++) {
        var matrix = get_network_weight(network_pointer, i);
        var matrix_columns_length = get_columns_length_from_matrix(matrix);

        if (network_config.weigths[i].length !== matrix_columns_length) {
            throw new Error('!weigths[i].length');
        }

        for (let j = 0; j < matrix_columns_length; j++) {
            var column_pointer = get_column_from_matrix(matrix, j);
            var row_length = get_elements_length_from_vector(column_pointer);
            var vector_elemnets_pointer = get_elements_from_vector(column_pointer);

            if (network_config.weigths[i][j].length !== row_length) {
                throw new Error('!weigths[i][j].length');
            }

            for (let w = 0; w < network_config.weigths[i][j].length; w++) {
                var value = getValue(vector_elemnets_pointer + (w * float_type_size), 'float');

                if (network_config.weigths[i][j][w] !== value) {
                    throw new Error('!value');
                }
            }
        }
    }


    for (var i = 0; i < network_config.sizes.length; i++) {
        console.log(`${i} size: ${getValue(_sizes + (i * int_type_size), 'i32')}`);
    }
}