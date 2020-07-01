var express = require('express');
var fs = require('fs');
var util = require('util');
var app = express();

var readFile = util.promisify(fs.readFile);

app.use(express.static(__dirname));

app.get('/', function (req, res) {
  readFile('./web/main.html').then(function (buffer) {
    res.contentType('html').send(buffer);
  });
});

app.listen(8080, function () {
  console.log('Example app listening on port 8080!');
});
