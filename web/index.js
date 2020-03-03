console.log(Module);

Module['onRuntimeInitialized'] = function () {
    var malloc = Module.cwrap('malloc', 'number', ['number']);
    var int_type_size = getNativeTypeSize('i32');

    var sizes = malloc(int_type_size);
    setValue(sizes + (int_type_size * 0), 784, 'i32');
    setValue(sizes + (int_type_size * 1), 30, 'i32');
    setValue(sizes + (int_type_size * 2), 10, 'i32');

    var network_pointer = Module.ccall('create_network', 'number', ['number', 'number'], [sizes, 3]);
    var _sizes = Module.ccall('get_network_sizes', 'number', ['number'], [network_pointer]);

    for (var i = 0; i < 3; i++) {
        console.log(`${i} size: ${getValue(_sizes + (i * int_type_size), 'i32')}`);
    }
}