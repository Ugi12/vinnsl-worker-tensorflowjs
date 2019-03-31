'use strict';

const express = require('express');
const createError = require('http-errors');
const exphbs  = require('express-handlebars');
const path = require('path');
const cookieParser = require('cookie-parser');
const logger = require('morgan');
const workerController = require('./public/nnworker/service/worker-controller');

// Constants
const PORT = 3000;
const HOST = '0.0.0.0';

// App
const app = express();
app.engine('.hbs', exphbs({extname: '.hbs'}));
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', '.hbs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/worker', workerController);

/*
app.get('/', (req, res) => {
    res.send('Hello world\n');
});
*/

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

app.listen(PORT, HOST);
console.log(`Running on http://${HOST}:${PORT}`);