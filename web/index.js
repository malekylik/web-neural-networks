var SHADER_TYPES = {
    unknown: -1,
    vertexShader: 0,
    fragmentShader: 1,
};

const DISABLED_PROGRAM = false;
const ENABLED_PROGRAM = true;

var IMAGES_NUMBER_OFFEST = 4;
var ROWS_NUMBER_OFFSET = 8;
var COLUMNS_NUMBER_OFFSET = 12;
var IMAGES_START_PIXEL_OFFSET = 16;

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
		var temp = images_buffer[IMAGES_START_PIXEL_OFFSET + i];
		var temp2 = temp / 256.0;
		images_buffer_f[IMAGES_START_PIXEL_OFFSET + i] = temp2;
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
	return data.images_buffer[IMAGES_START_PIXEL_OFFSET + (image_number * data.rows_number * data.columns_number) + pixel_offset];
}

function get_pixel_of_image_by_cord(data, image_number, x, y) {
	return data.images_buffer[IMAGES_START_PIXEL_OFFSET + (image_number * data.rows_number * data.columns_number) + (data.columns_number * y) + x];
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
var input_data_pointer = null;
var image_data  = new ImageData(24 * scale_image, 24 * scale_image);
var input = document.getElementsByClassName('image_number_input')[0];
var image_indicator = document.getElementsByClassName('images-indicator')[0];
var image_real_answer = document.getElementsByClassName('images-real-answer')[0];
var image_network_answer = document.getElementsByClassName('images-network-answer')[0];
var image_network_answer = document.getElementsByClassName('images-network-answer')[0];


function update_image_data(mnist_data, image_number) {
    for (let i = 0; i < 24 * scale_image; i++) {
        for (let j = 0; j < 24 * scale_image; j++) {
            // var temp = get_pixel_of_image_by_cord(mnist_data, image_number, 24 - ((j / scale_image) | 0), ((i / scale_image) | 0));
            var temp = get_pixel_of_image_by_cord(mnist_data, image_number, ((j / scale_image) | 0), ((i / scale_image) | 0));
            putPixel(image_data, i, j, temp);
        }
    }
}

function createGLShader(gl, type, shaderSrc) {
    const glType = getGLShaderType(gl, type);

    if (glType === SHADER_TYPES.unknown) console.warn(`Unknown shader type: ${type}`);

    const shader = gl.createShader(glType);

    if (shader === 0) console.warn('Fail to create shader');

    gl.shaderSource(shader, shaderSrc);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader);

      console.warn(`Fail to compile shader: ${message}`);

      gl.deleteShader(shader);

      return null;
    }

    return ({
        type,
        nativeTypy: glType,
        nativeShader: shader,
        source: shaderSrc,
        isCompiled: true,
        isDeleted: false,
    });
}

function getGLShaderType(gl, type) {
    switch (type) {
      case SHADER_TYPES.vertexShader: return gl.VERTEX_SHADER;
      case SHADER_TYPES.fragmentShader: return gl.FRAGMENT_SHADER;
    }
  
    return SHADER_TYPES.unknown;
}

function deleteShader(gl, shader) {
    gl.deleteShader(getNativeShader(shader));
    setIsDelete(shader, true);
  }
  

function getShaderType(shader) {
    return shader.type;
  }
  

function getNativeShader(shader) {
    return shader.nativeShader;
  }
  
function setIsDelete(shader, value) {
    return shader.isDeleted = value;
}

function createGLProgram(gl, vertShader, fragShader) {
    const program = gl.createProgram();
  
    if (getShaderType(vertShader) !== SHADER_TYPES.vertexShader) console.warn('Invalid vertex shader type');
    if (getShaderType(fragShader) !== SHADER_TYPES.fragmentShader) console.warn('Invalid fragment shader type');
  
    gl.attachShader(program, getNativeShader(vertShader));
    gl.attachShader(program, getNativeShader(fragShader));
  
    gl.linkProgram(program);
  
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
  
      console.warn(`Fail to link program: ${info}`);
  
      gl.deleteProgram(program);
  
      return null;
    }
  
    return createProgram(program, true, false, false);
  }
  
function createProgram(nativeProgram, isLinked, isUsed, isDeleted) {
    return ({
      nativeProgram,
      isLinked,
      isUsed,
      isDeleted,
    });
  }
  
function getNativeProgram(program) {
    return program.nativeProgram;
  }
  
function useProgram(gl, program) {
    gl.useProgram(getNativeProgram(program));
    setProgramUsed(program, ENABLED_PROGRAM);
  }
  
function validateProgram(gl, program) {
    const nativeProgram = getNativeProgram(program);
  
    gl.validateProgram(nativeProgram);
  
    if (!gl.getProgramParameter(nativeProgram, gl.VALIDATE_STATUS)) {
      return gl.getProgramInfoLog(nativeProgram);
    }
  
    return '';
  }
  
  function setProgramUsed(program, value) {
    return program.isUsed = value;
  }

function normolizeToGLX(x, width) {
  return (x / width * 2) - 1;
}

function normolizeToGLY(y, height) {
  return 1 - (y / height * 2);
}

var vert = `\
#version 300 es

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec2 vCenter;

// out vec2 vCenter;

void main()
{
  gl_Position = vec4(vPosition, 1.0);
}
`;

var frag = `\
#version 300 es

precision mediump float;

in vec2 vCenter;
out vec4 fragColor;

void main()
{
  fragColor = vec4(gl_FragCoord.x / 480.0, 0.0, 0.0, 1.0);
}
`;

// var line_stride = Float32Array.BYTES_PER_ELEMENT * 5;
var elems_per_line = 3 * 3 * 2;
var max_lines_count = 1000;
var max_lines_size = max_lines_count * elems_per_line;
var square_size = 30;
var half_square_size = square_size / 2;

var current_lines_count = 0;

var lines_coords = new Float32Array(max_lines_size);

var canvas_width = draw_digit_canvas.width;
var canvas_height = draw_digit_canvas.height;

var prev_point = {
  x: -2,
  y: -2,
  z: -2,
};

var clear_request_frame = 0;

console.log('canvas_width', canvas_width);

var pixels = new Float32Array(canvas_width * canvas_height * 3);

Module['onRuntimeInitialized'] = async function () {
    var gl = draw_digit_canvas.getContext('webgl2');
    gl.viewport(0, 0, canvas_width, canvas_height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    var vertShader = createGLShader(gl, SHADER_TYPES.vertexShader, vert);
    var fragShader = createGLShader(gl, SHADER_TYPES.fragmentShader, frag);

    const program = createGLProgram(gl, vertShader, fragShader);

    deleteShader(gl, vertShader);
    deleteShader(gl, fragShader);

    var pointsBuffer = gl.createBuffer();
    // var framebuffer = gl.createFramebuffer();

    useProgram(gl, program);

    gl.bindBuffer(gl.ARRAY_BUFFER, pointsBuffer);
    // gl.bufferData(gl.ARRAY_BUFFER, lines_coords, gl.STATIC_DRAW);
    gl.bufferData(gl.ARRAY_BUFFER, lines_coords, gl.DYNAMIC_DRAW);
    
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, Float32Array.BYTES_PER_ELEMENT * 3, 0);
    // gl.vertexAttribPointer(0, 2, gl.FLOAT, false, Float32Array.BYTES_PER_ELEMENT * 5, Float32Array.BYTES_PER_ELEMENT * 3);
    gl.enableVertexAttribArray(0);
    // gl.enableVertexAttribArray(1);

    function drawNewLines() {
      gl.clear(gl.COLOR_BUFFER_BIT);

      // console.log('current_lines_count * 2', current_lines_count * 2);
  
      // gl.drawArrays(gl.TRIANLES, 0, current_lines_count * 6);
      gl.drawArrays(gl.TRIANGLES, 0, current_lines_count * 6);
    }

    drawNewLines();

    draw_digit_canvas.addEventListener('mousedown', function (e) {
      function mouse_move (e) {
        if (max_lines_count <= current_lines_count) return;
        // if (10 <= current_lines_count) return;

        var offsetX = e.offsetX;
        var offsetY = e.offsetY;

        if (prev_point.x < -1) {
          // prev_point.x = (offsetX / canvas_width * 2) - 1;
          // prev_point.y = 1 - (offsetY / canvas_height * 2);
          // prev_point.z = 1.0;
          prev_point.x = offsetX;
          prev_point.y = offsetY;
          prev_point.z = 1.0;
        } else {
          var prev_top_ver = normolizeToGLY(prev_point.y - half_square_size, canvas_height);
          var prev_bottom_ver = normolizeToGLY(prev_point.y + half_square_size, canvas_height);
          var prev_left_ver = normolizeToGLX(prev_point.x - half_square_size, canvas_width);

          var next_top_ver = normolizeToGLY(offsetY - half_square_size, canvas_height);
          var next_right_ver = normolizeToGLX(offsetX + half_square_size, canvas_width);
          var next_bottom_ver = normolizeToGLY(offsetY + half_square_size, canvas_height);

          // 1 face
          // top - left
          lines_coords[current_lines_count * elems_per_line + 0] = prev_left_ver;
          lines_coords[current_lines_count * elems_per_line + 1] = prev_top_ver;
          lines_coords[current_lines_count * elems_per_line + 2] = prev_point.z;

          // top - right
          lines_coords[current_lines_count * elems_per_line + 3] = next_right_ver;
          lines_coords[current_lines_count * elems_per_line + 4] = next_top_ver;
          lines_coords[current_lines_count * elems_per_line + 5] = prev_point.z;

          // bottom - left
          lines_coords[current_lines_count * elems_per_line + 6] = prev_left_ver;
          lines_coords[current_lines_count * elems_per_line + 7] = prev_bottom_ver;
          lines_coords[current_lines_count * elems_per_line + 8] = prev_point.z;


          // // prev center
          // lines_coords[current_lines_count * elems_per_line + 9] = normolizeToGLX(prev_point.x, canvas_width);
          // lines_coords[current_lines_count * elems_per_line + 10] = normolizeToGLY(prev_point.y, canvas_height);

          // 2 face
          // bottom - left
          lines_coords[current_lines_count * elems_per_line + 9] = prev_left_ver;
          lines_coords[current_lines_count * elems_per_line + 10] = prev_bottom_ver;
          lines_coords[current_lines_count * elems_per_line + 11] = prev_point.z;

          // bottom - right
          lines_coords[current_lines_count * elems_per_line + 12] = next_right_ver;
          lines_coords[current_lines_count * elems_per_line + 13] = next_bottom_ver;
          lines_coords[current_lines_count * elems_per_line + 14] = prev_point.z;

          // top - right
          lines_coords[current_lines_count * elems_per_line + 15] = next_right_ver;
          lines_coords[current_lines_count * elems_per_line + 16] = next_top_ver;
          lines_coords[current_lines_count * elems_per_line + 17] = prev_point.z;

          // // next center
          // lines_coords[current_lines_count * elems_per_line + 20] = normolizeToGLX(prev_point.x, canvas_width);
          // lines_coords[current_lines_count * elems_per_line + 21] = normolizeToGLY(prev_point.y, canvas_height);

          prev_point.x = offsetX;
          prev_point.y = offsetY;

          // prev_point.x = lines_coords[current_lines_count * elems_per_line + 3] = (offsetX / canvas_width * 2) - 1;
          // prev_point.y = lines_coords[current_lines_count * elems_per_line + 4] = 1 - (offsetY / canvas_height * 2);
          // prev_point.z = lines_coords[current_lines_count * elems_per_line + 5] = 1.0;

          // console.log('lines_coords', lines_coords);
          current_lines_count += 1;

          console.log('clear_request_frame', clear_request_frame);
          if (clear_request_frame === 0) {

            clear_request_frame = requestAnimationFrame(function () {
              gl.bufferData(gl.ARRAY_BUFFER, lines_coords, gl.DYNAMIC_DRAW);
              
              console.log('current_lines_count', current_lines_count);
              
              // gl.bindFramebuffer(gl.FRAMEBUFFER, 0);
              // drawNewLines();
              
              // gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
              // gl.viewport(0, 0, canvas_width, canvas_height);
              drawNewLines();

              // gl.readPixels(0, 0, canvas_width, canvas_height, gl.RGB, gl.FLOAT, pixels);
              

              // console.log(pixels);

              clear_request_frame = 0;
            });
          }
        }


        // lines_coords[current_lines_count * 2 + 0] = lines_coords[current_lines_count * 2 - 3];
        // lines_coords[current_lines_count * 2 + 1] = lines_coords[current_lines_count * 2 - 2];
        // lines_coords[current_lines_count * 2 + 2] = lines_coords[current_lines_count * 2 - 1];

        // lines_coords[current_lines_count * elems_per_line + 3] = (offsetX / canvas_width * 2) - 1;
        // lines_coords[current_lines_count * elems_per_line + 4] = 1 - (offsetY / canvas_height * 2);
        // lines_coords[current_lines_count * elems_per_line + 5] = 1.0;


        // var prevX = lines_coords[current_lines_count - 3];
        // var prevY = lines_coords[current_lines_count - 2];
        // var prevZ = lines_coords[current_lines_count - 1];

        // var newX = (clientX / canvas_width / 2) - 1;
        // var newY = 1 - (clientY / canvas_height / 2)
        // var newZ = 1.0;

        // lines_coords.push(
        //   prevX, prevY, prevZ,
        //   newX, newY, newZ
        // );
      }

      function mouse_up (e) {
        draw_digit_canvas.removeEventListener('mousemove', mouse_move);
        draw_digit_canvas.removeEventListener('mouseup', mouse_up);

        prev_point.x = -2;
      }

      draw_digit_canvas.addEventListener('mousemove', mouse_move);
      draw_digit_canvas.addEventListener('mouseup', mouse_up);


      // console.log('down e', e);
    });


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
    var set_vector_value = Module.cwrap('set_vector_value', 'void', ['number', 'number', 'number']);
    var get_max_value_index_in_vector = Module.cwrap('get_max_value_index_in_vector', 'number', ['number']);
    var feedforward = Module.cwrap('feedforward', 'number', ['number', 'number']);
    var create_vector = Module.cwrap('create_vector', 'number', ['number']);
    var malloc = Module.cwrap('malloc', 'number', ['number']);
    var free = Module.cwrap('free', 'void', ['number']);

    function fill_input_data(data, image_number, input_data_pointer, length) {
        for (let i = 0; i < length; i++) {
            set_vector_value(input_data_pointer, i, get_pixel_of_image(data, image_number, i));
        }
    }

    console.log('config', network_config);

    var int_type_size = getNativeTypeSize('i32');

    var mnist_data = create_mnist_data(images, labels);
    image_indicator.innerText = mnist_data.image_number;

    var sizes_pointer = malloc(int_type_size * network_config.sizes.length);

    network_config.sizes.forEach((size, i) => setValue(sizes_pointer + (int_type_size * i), size, 'i32'));

    var network_pointer = Module.ccall('create_network', 'number', ['number', 'number'], [sizes_pointer, 3]);
    input_data_pointer = create_vector(784);

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

    fill_input_data(mnist_data, 0, input_data_pointer, 784);
    var vector_answer = feedforward(network_pointer, input_data_pointer);
    var answer = get_max_value_index_in_vector(vector_answer);
    free(vector_answer);
    image_network_answer.innerText = answer;
    image_real_answer.innerText = get_label_for_image(mnist_data, 0);

    input.addEventListener('change', () => {
        var image_number = Number(input.value);

        update_image_data(mnist_data, image_number);
        ctx.putImageData(image_data, 0, 0);
        fill_input_data(mnist_data, image_number, input_data_pointer, 784);
        var vector_answer = feedforward(network_pointer, input_data_pointer);
        var answer = get_max_value_index_in_vector(vector_answer);
        free(vector_answer);

        image_network_answer.innerText = answer;
        image_real_answer.innerText = get_label_for_image(mnist_data, image_number);
    });

    var ctx = digit_canvas.getContext('2d');
    update_image_data(mnist_data, 0);
    ctx.putImageData(image_data, 0, 0);
}