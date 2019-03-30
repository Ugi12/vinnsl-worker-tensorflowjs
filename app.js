var createError = require('http-errors');
var express = require('express');
var exphbs  = require('express-handlebars');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

//var bodyParser = require('body-parser');
//var multer = require('multer');
//var upload = multer();


var workerController = require('./public/nnworker/service/worker-controller');

var app = express();

// view engine setup
app.engine('.hbs', exphbs({extname: '.hbs'}));
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', '.hbs');


//app.use(bodyParser({limit: '50mb'}));


app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));


// for parsing application/json
//app.use(bodyParser.json());

// for parsing application/xwww-
//app.use(bodyParser.urlencoded({ extended: true }));
//form-urlencoded

// for parsing multipart/form-data
//app.use(upload.array());


app.use('/worker', workerController);
//app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};
  //res.setHeader('Access-Control-Allow-Origin', '*');
  //res.setHeader("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});


module.exports = app;
