let request = require('request');
let mnist = require('../../util/mnist/mnist');
let dateFormat = require('dateformat');

const axios = require('axios');
const addresses = require('../../util/addresses');
const nnStatus = require('../../util/nn-status');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');


let convert = require('xml-js');


const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/mnist';

let TRAINING_DURATION = 0;

//default values
const FEATURES_COUNT = 4;
const CLASSES_COUNT = 3;
const HIDDEN_COUNT = 3;
const LABEL_INDEX = 0;
const INPUT_NEURONS = 0;
const OUTPUT_NEURONS = 0;
const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_EPOCHS = 20;
const DEFAULT_BATCH_SIZE = 128;
const DEFAULT_ACTIVATION_FUNCTON = 'relu'



let learningRate = null;

let labelIndex = null;
let schemaID = null;


let options = {
    compact: true,
    spaces: 2,
    ignoreDeclaration: true,
    ignoreInstruction: true,
    ignoreAttributes: false,
    ignoreComment: true,
    ignoreCdata: true,
    ignoreDoctype: true

};

let inputNeurons = null;
let outputNeurons = null;
let hiddenlayers = null;
let hiddenNeurons = [];
let activationFunction = null;
let epochs = null;
let batchSize = null;



//const data = require('./data');
//const model = require('./model');




module.exports = {

    mnistTrainer: async function (id) {
        //mnistTrainer: async function (id, epochs, batchSize) {
        try {
            const validationSplit = 0.15;
            let trainBatchCount = 0;
            let valAcc;

            /**
             * Get MNIST DATA as Tensors
             */
            const [trainImages, trainLabels, testImages, testLabels] = await mnist.getData();

            let startTime = Date.now();
            if(fs.existsSync(FILE_SAVE_PATH +'/'+ id +'/model.json')){
                console.log('model exist... load model...');

                request(addresses.getVinnslServiceEndpoint() + '/vinnsl/' + id,  async function (error, response, body) {
                    try {
                        let json = convert.xml2json(body, options);
                        let vinnslJSON = JSON.parse(json);
                        activationFunction = getActivationFunction(vinnslJSON);
                        epochs = getIterations(vinnslJSON);
                        batchSize = getBatchSize(vinnslJSON);



                        /**
                         *
                         * load model
                         */
                        const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH +'/' + id +'/model.json'}`);

                        /**
                         * Compile model
                         */
                        const optimizer = 'rmsprop';
                        model.compile({
                            optimizer: optimizer,
                            loss: 'categoricalCrossentropy',
                            metrics: ['accuracy']
                        });


                        const numTrainExamplesPerEpoch =
                            trainImages.shape[0] * (1 - validationSplit);
                        const numTrainBatchesPerEpoch =
                            Math.ceil(numTrainExamplesPerEpoch / batchSize);
                        const totalNumTrainBatches = numTrainBatchesPerEpoch * epochs;
                        const totalNumTrainBatchesInPercent = (100 / totalNumTrainBatches).toFixed(2);
                        let trainingProcess = 0;

                        /**
                         * Train model
                         */
                        await model.fit(trainImages, trainLabels,{
                            epochs,
                            batchSize,
                            validationSplit,

                            callbacks:{
                                onBatchEnd: async (batch, logs) => {
                                    if(trainBatchCount % 10 == 0){
                                        trainingProcess = (totalNumTrainBatchesInPercent * trainBatchCount).toFixed(1);
                                        createOrUpdateTraningProcess(id, trainingProcess);
                                    }
                                    await tf.nextFrame();
                                },
                                onBatchBegin: async (epoch) => {
                                    console.log('batch nr: '+ ++trainBatchCount)
                                },
                                onEpochBegin: async (epoch) =>{
                                    console.log('epoch: '+ (epoch+1) +'/'+epochs+' started.')
                                },
                                onTrainBegin: async ()=>{
                                    console.log('training started. '+new Date())
                                    createOrUpdateTraningProcess(id, 0);
                                },
                                onTrainEnd: async ()=>{
                                    console.log('training end. ' +new Date())
                                    nnStatus.setStatusToFinished(id);
                                    createOrUpdateTraningProcess(id, 100);
                                    await model.save(`file://${FILE_SAVE_PATH}/` + id);
                                }
                            }
                        });

                        const evalOutput = model.evaluate(testImages, testLabels);

                        let endTime = Date.now();
                        TRAINING_DURATION = endTime-startTime;
                        TRAINING_DURATION = (TRAINING_DURATION/1000/60).toFixed(0);

                        let predictionInPercent = (evalOutput[1].dataSync()[0].toFixed(4)) * 100;
                        let loss = evalOutput[0].dataSync()[0].toFixed(3);


                        createStatistics(TRAINING_DURATION, predictionInPercent.toFixed(), loss, epochs, batchSize, id);


                        console.log(
                            `\nEvaluation result:\n` +
                            `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
                            `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

                    }catch (e) {
                        console.log(e);
                    }
                });



            }else{

                /**
                 * Model not exist..
                 * Read data from VINNSL definition and create/train model
                 */
                request(addresses.getVinnslServiceEndpoint() + '/vinnsl/' + id,  async function (error, response, body) {
                    try{
                        let json = convert.xml2json(body, options);
                        let vinnslJSON = JSON.parse(json);
                        inputNeurons = getinputNeurons(vinnslJSON);
                        outputNeurons = getOutputNeurons(vinnslJSON);
                        activationFunction = getActivationFunction(vinnslJSON);
                        epochs = getIterations(vinnslJSON);
                        batchSize = getBatchSize(vinnslJSON);
                        getHiddenCount(vinnslJSON);
                        labelIndex = getLabelIndex(vinnslJSON);
                        schemaID = getDataSchemaID(vinnslJSON);



                        /**
                         * Create Model
                         */
                        let model =  await createModel(id);


                        const numTrainExamplesPerEpoch =
                            trainImages.shape[0] * (1 - validationSplit);
                        const numTrainBatchesPerEpoch =
                            Math.ceil(numTrainExamplesPerEpoch / batchSize);
                        const totalNumTrainBatches = numTrainBatchesPerEpoch * epochs;
                        const totalNumTrainBatchesInPercent = (100 / totalNumTrainBatches).toFixed(2);
                        let trainingProcess = 0;
                        /**
                         * Train model
                         */
                        await model.fit(trainImages, trainLabels,{
                            epochs,
                            batchSize,
                            validationSplit,

                            callbacks:{
                                onBatchEnd: async (batch, logs) => {

                                    if(trainBatchCount % 10 == 0){
                                        trainingProcess = (totalNumTrainBatchesInPercent * trainBatchCount).toFixed(1);
                                        createOrUpdateTraningProcess(id, trainingProcess);
                                    }

                                    await tf.nextFrame();
                                },
                                onBatchBegin: async (epoch) => {
                                    console.log('batch nr: '+ ++trainBatchCount)
                                },
                                onEpochBegin: async (epoch) =>{
                                    console.log('epoch: '+ (epoch+1) +'/'+epochs+' started.')
                                },
                                onTrainBegin: async ()=>{
                                    console.log('training started. '+new Date());
                                    createOrUpdateTraningProcess(id, 0);
                                },
                                onTrainEnd: async ()=>{
                                    console.log('training end. ' +new Date())
                                    nnStatus.setStatusToFinished(id);
                                    createOrUpdateTraningProcess(id, 100);
                                    await model.save(`file://${FILE_SAVE_PATH}/` + id);
                                }
                            }
                        });

                        const evalOutput = model.evaluate(testImages, testLabels);


                        let endTime = Date.now();
                        TRAINING_DURATION = endTime-startTime;
                        TRAINING_DURATION = (TRAINING_DURATION/1000/60).toFixed(0);

                        let predictionInPercent = (evalOutput[1].dataSync()[0].toFixed(4)) * 100;
                        let loss = evalOutput[0].dataSync()[0].toFixed(3);

                        createStatistics(TRAINING_DURATION, predictionInPercent.toFixed(2), loss, epochs, batchSize, id);

                        console.log(
                            `\nEvaluation result:\n` +
                            `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
                            `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(4)}`);


                    }catch (e) {
                        console.log(e);
                    }

                });


            }
        }catch (err) {
            console.log(err);
        }

    },function(err){
        console.log(err);
    }

};

function createStatistics(trainingTime, bestResult, loss, epochs, batchSize, id) {

    axios.post(addresses.getVinnslServiceEndpoint() + '/vinnsl/create-update/statistic', {
        id: id,
        createTimestamp: dateFormat(new Date(), "UTC:dd.mm.yyyy hh:MM:ss TT"),
        trainingTime: trainingTime ,
        numberOfTraining: 1,
        lastResult: bestResult,
        bestResult: bestResult,
        epochs: epochs,
        loss: loss,
        batchSize: batchSize

    })
    .then(function (response) {
        //ignore response
    })
    .catch(function (error) {
        console.log(error);
    });
}

function createOrUpdateTraningProcess(id, trainingInPercent) {

    axios.post(addresses.getVinnslServiceEndpoint() + '/vinnsl/create-update/process', {
        id: id,
        trainingProcess: trainingInPercent
    })
    .then(function (response) {
        //ignore response
    })
    .catch(function (error) {
        console.log(error);
    });

}


function createModel(id){



    //if an error occurs while reading vinnsl definition, break forEach loop and use default model.
    let useDefaultModel = false;

    const model = tf.sequential();

    //convolutional layer 1
    model.add(tf.layers.conv2d({
        inputShape: [28,28,1],
        kernelSize: 3,
        activation: 'relu',
        filters: 32
    }));

    hiddenNeurons.forEach(function (value, index, array) {

        if(value[0] === '' || typeof value[0] === 'undefined' || value[0] === null ||
           value[1] === '' || typeof value[1] === 'undefined' || value[1] === null) {
            useDefaultModel = true;
        }


        if(!useDefaultModel){

            if(value[0].toLowerCase().includes('conv')){
                model.add(tf.layers.conv2d({
                    kernelSize: 3,
                    activation: 'relu',
                    filters: value[1]
                }));

            } else if(value[0].toLowerCase().includes('maxpool')){
                model.add(tf.layers.maxPooling2d({
                    poolSize: [2, 2]
                }));

            } else if(value[0].toLowerCase().includes('flat')){
                model.add(tf.layers.flatten());

            } else if(value[0].toLowerCase().includes('dropout')){
                let rate = (value[1] / 100);
                model.add(tf.layers.dropout({
                    rate: rate
                }));

            } else if(value[0].toLowerCase().includes('dense')){
                model.add(tf.layers.dense({
                    units: value[1],
                    activation: 'relu'
                }));
            }
        }
    });

    if(!useDefaultModel){
        //output layer
        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));

        const optimizer = 'rmsprop';

        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        model.summary();

        return model;

    } else {


    /**
     * Default model
     */
        const model = tf.sequential();

        //convolutional layer 1
        model.add(tf.layers.conv2d({
            inputShape: [28,28,1],
            kernelSize: 3,
            activation: 'relu',
            filters: 32
        }));

        //convolutional layer 2
        model.add(tf.layers.conv2d({
            kernelSize: 3,
            activation: 'relu',
            filters: 32
        }));

        //max pooling layer 1
        model.add(tf.layers.maxPooling2d({
            poolSize: [2, 2]
        }));

        //convolutional layer 3
        model.add(tf.layers.conv2d({
            kernelSize: 3,
            activation: 'relu',
            filters: 64
        }));
        //convolutional layer 4
        model.add(tf.layers.conv2d({
            kernelSize: 3,
            activation: 'relu',
            filters: 64
        }));

        //max pooling layer 2
        model.add(tf.layers.maxPooling2d({
            poolSize: [2, 2]
        }));

        model.add(tf.layers.flatten());

        model.add(tf.layers.dropout({
            rate: 0.25
        }));

        model.add(tf.layers.dense({
            units: 512,
            activation: 'relu'
        }));

        model.add(tf.layers.dropout({
            rate: 0.5
        }));

        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));

        const optimizer = 'rmsprop';

        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        model.summary();


        return model;
    }



}


async function trainModel(xTrain, yTrain, xTest, yTest) {



    return;
}





/**
 * get the number of input neurons
 * @param vinnslJSON
 * @returns {number} of input neurons
 */
function getinputNeurons(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.structure){
                    if(vinnslJSON.vinnsl.definition.structure.input){
                        if(vinnslJSON.vinnsl.definition.structure.input.size){
                            if(vinnslJSON.vinnsl.definition.structure.input.size._text > 0) {
                                return parseInt(vinnslJSON.vinnsl.definition.structure.input.size._text,10);
                            }
                        }
                    }
                }
            }
        }
    }
    return INPUT_NEURONS;
}

/**
 * get the number of output neurons
 * @param vinnslJSON
 * @returns {number} of output neurons
 */
function getOutputNeurons(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.structure){
                    if(vinnslJSON.vinnsl.definition.structure.output){
                        if(vinnslJSON.vinnsl.definition.structure.output.size){
                            if(vinnslJSON.vinnsl.definition.structure.output.size._text > 0) {
                                return parseInt(vinnslJSON.vinnsl.definition.structure.output.size._text,10);
                            }
                        }
                    }
                }
            }
        }
    }
    return OUTPUT_NEURONS;
}



function getHiddenCount(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.structure){
                    if(vinnslJSON.vinnsl.definition.structure.hidden){
                        if(vinnslJSON.vinnsl.definition.structure.hidden){
                            let obj = vinnslJSON.vinnsl.definition.structure.hidden;
                            if(typeof obj.length === 'undefined'  && parseInt(obj.size._text) > 0){
                                hiddenlayers = 1;
                                let tempArray = [];
                                tempArray.push(vinnslJSON.vinnsl.definition.structure.hidden.ID._text);
                                tempArray.push(parseInt(vinnslJSON.vinnsl.definition.structure.hidden.size._text));
                                hiddenNeurons.push(tempArray);

                            }else if(Array.isArray(obj) && obj.length > 0){
                                hiddenlayers = vinnslJSON.vinnsl.definition.structure.hidden.length;
                                for(i in vinnslJSON.vinnsl.definition.structure.hidden){
                                    let tempArray = [];
                                    tempArray.push(vinnslJSON.vinnsl.definition.structure.hidden[i].ID._text);
                                    tempArray.push(parseInt(vinnslJSON.vinnsl.definition.structure.hidden[i].size._text, 10));
                                    hiddenNeurons.push(tempArray);
                                    //hiddenNeurons.push(parseInt(vinnslJSON.vinnsl.definition.structure.hidden[i].size._text,10));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

function getLabelIndex(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.parameters){
                    if(vinnslJSON.vinnsl.definition.parameters.valueparameter){
                        for(i in vinnslJSON.vinnsl.definition.parameters.valueparameter){
                            if(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._attributes.name === 'labelIndex'){
                                return parseInt(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._text,10);
                            }
                        }
                    }
                }
            }
        }
    }
    return LABEL_INDEX;
}

/**
 *
 * @param vinnslJSON
 * @returns {number} of learning rate
 */
function getLearningRate(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.parameters){
                    if(vinnslJSON.vinnsl.definition.parameters.valueparameter){
                        for(i in vinnslJSON.vinnsl.definition.parameters.valueparameter){
                            if(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._attributes.name === 'learningrate'){
                                return parseFloat(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._text);
                            }
                        }
                    }
                }
            }
        }
    }
    return DEFAULT_LEARNING_RATE;
}

/**
 *
 * @param vinnslJSON
 * @returns {number} of iterations (epochs)
 */
function getIterations(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.parameters){
                    if(vinnslJSON.vinnsl.definition.parameters.valueparameter){
                        for(i in vinnslJSON.vinnsl.definition.parameters.valueparameter){
                            if(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._attributes.name === 'epochs'){
                                return parseFloat(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._text);
                            }
                        }
                    }
                }
            }
        }
    }
    return DEFAULT_EPOCHS;
}

/**
 *
 * @param vinnslJSON
 * @returns {number} of batchsize
 */
function getBatchSize(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.parameters){
                    if(vinnslJSON.vinnsl.definition.parameters.valueparameter){
                        for(i in vinnslJSON.vinnsl.definition.parameters.valueparameter){
                            if(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._attributes.name === 'batchsize'){
                                return parseFloat(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._text);
                            }
                        }
                    }
                }
            }
        }
    }
    return DEFAULT_BATCH_SIZE;
}

/**
 *
 * @param vinnslJSON
 * @returns name of the activation function
 */
function getActivationFunction(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.parameters){
                    if(vinnslJSON.vinnsl.definition.parameters.comboparameter){
                        for(i in vinnslJSON.vinnsl.definition.parameters.comboparameter){
                            if(vinnslJSON.vinnsl.definition.parameters.comboparameter[i]._attributes.name === 'activationfunction'){
                                return vinnslJSON.vinnsl.definition.parameters.comboparameter[i]._text;
                            }
                         }
                    }
                }
            }
        }
    }
    return DEFAULT_ACTIVATION_FUNCTON;
}


/**
 * Get the name of the file e.g. (mnist.txt)
 * @param vinnslJSON
 * @returns {name} of the file, if not found return null
 */
function getDataSchemaID(vinnslJSON) {
    if(vinnslJSON) {
        if (vinnslJSON.vinnsl) {
            if (vinnslJSON.vinnsl.definition) {
                if (vinnslJSON.vinnsl.definition.data) {
                    if (vinnslJSON.vinnsl.definition.data.dataSchemaID) {
                        return vinnslJSON.vinnsl.definition.data.dataSchemaID._text;
                    }
                }
            }
        }
    }
    return null;
}



