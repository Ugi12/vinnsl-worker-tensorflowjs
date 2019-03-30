const express = require('express');

const tf = require('@tensorflow/tfjs-node');

Stream = require('stream').Transform;

const axios = require('axios');


const fs = require('fs');
const tf_iris_network_trainer = require('../tensorflow/iris/network-trainer');
const tf_mnist_network_trainer = require('../tensorflow/mnist/network-trainer');
const tf_mnist_test_model = require('../tensorflow/mnist/test-model');
const tf_wine_network_trainer = require('../tensorflow/wine/network-trainer');
const tf_lstm_network_trainer = require('../tensorflow/lstm/network-trainer');
const tf_lstm_text_generation = require('../tensorflow/lstm/text-generation');
const addresses = require('../util/addresses');
const nnStatus = require('../util/nn-status')


const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/mnist';


var router = express.Router();



/**
 * API endpoint for IRIS classification with TensorFlow-JS
 */
router.post('/iris', function (req, res, next) {

    try {
        let id = req.body.id;
        let epochs = req.body.epochs;
        let learningRate = req.body.learningRate;
        let vinnslItem = req.body.vinnslItem;

        nnStatus.setStatusToInProgress(id)
        tf_iris_network_trainer.irisTrainer(id, vinnslItem, epochs, learningRate, res);
    }catch (e) {
        console.log(e);
    }

});


/**
 * API endpoint for MNIST classification with TensorFlow-JS
 */
router.post('/mnist', function (req, res, next) {

    try {
        let id = req.body.id;
        let epochs = req.body.epochs;
        let batchSize = req.body.batchSize;

        nnStatus.setStatusToInProgress(id);
        tf_mnist_network_trainer.mnistTrainer(id, parseInt(epochs), parseInt(batchSize));
        res.send(null);
     }catch (e) {
        console.log(e);
    }

});


/**
 * API endpoint for WINE classification with TensorFlow-JS
 */
router.post('/wine', function (req, res, next) {

    try {
        let id = req.body.id;
        let epochs = req.body.epochs;
        let learningRate = req.body.learningRate;

        nnStatus.setStatusToInProgress(id);
        tf_wine_network_trainer.wineTrainer(id, epochs, learningRate);
        res.send(null);
    }catch (e) {
        console.log(e);
    }

});


/**
 * API endpoint for LSTM model with TensorFlow-JS
 */
router.post('/lstm', function (req, res, next) {

    try {
        let id = req.body.id;
        let text = req.body.text;


        //nnStatus.setStatusToInProgress(id);
        tf_lstm_network_trainer.lstmTrainer(id, text);
        res.send(null);
    }catch (e) {
        console.log(e);
    }

});



router.post('/predict-on-server/mnist', async function (req, res, next) {

    let imageData = req.body.imageData;

    let predictedValue = await tf_mnist_test_model.mnistTester(imageData);

    res.send(predictedValue);

});

/**
 * API endpoint for saving text for LSTM network
 */
router.post('/save/text/lstm', function (req, res, next) {

    try {
        let id = req.body.id;
        let text = req.body.text;
        // text = text.replace(/\\n/g, '');



        axios.post(addresses.getVinnslServiceEndpoint() + '/vinnsl/save-text/lstm', {
            id: id,
            text: text
        })
        .then(response => {
            let textIsChanged = response.data.textChanged;
        })
        .catch(error => {
            console.log(error);
        });


        //nnStatus.setStatusToInProgress(id);
        // tf_lstm_network_trainer.lstmTrainer(id, text);
        res.send(null);
    }catch (e) {
        console.log(e);
    }

});
/**
 * API endpoint for training LSTM network
 */
router.post('/lstm', function (req, res, next) {

    try {
        let id = req.body.id;
        // let text = req.body.text;
        nnStatus.setStatusToInProgress(id);
        tf_lstm_network_trainer.lstmTrainer(id);
        res.send(null);
    }catch (e) {
        console.log(e);
    }

});

/**
 * API endpoint for generate text for LSTM model with TensorFlow-JS
 */
router.post('/lstm/generate/text', async function (req, res, next) {

    try {
        let id = req.body.id;
        let textLen = parseInt(req.body.textLen);
        let temperature = parseFloat(req.body.temperature);


        const result = await tf_lstm_text_generation.lstmTextGeneration(id, textLen, temperature);
        res.send(result);
    }catch (e) {
        console.log(e);
    }

});



module.exports = router;