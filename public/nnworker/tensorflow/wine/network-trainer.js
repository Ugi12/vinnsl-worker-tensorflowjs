let request = require('request');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const converter = require('../../util/wine/converter')
let addresses = require('../../util/addresses');
const axios = require('axios');
let dateFormat = require('dateformat');

const wineJS = require('../../data/wine/wine');
const nnStatus = require('../../util/nn-status');

const convert = require('xml-js');
const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/wine';

const TEST_DATA_IN_PERCENT = 0.20;

//default values
const INPUT_NEURONS = 4;
const OUTPUT_NEURONS = 3;
const HIDDEN_LAYERS = 1;
const LABEL_INDEX = 0;

const DEFAULT_LEARNING_RATE = 0.001;
const DEFAULT_EPOCHS = 5000;
const DEFAULT_BATCH_SIZE = 100;
const DEFAULT_ACTIVATION_FUNCTON = 'relu'

let inputNeurons = null;
let outputNeurons = null;
let hiddenlayers = null;
let hiddenNeurons = [];
let activationFunction = null;
let epochs = null;
let batchSize = null;
let learningRate = null;
let labelIndex = null;
let schemaID = null;

async function trainModel(xTrain, yTrain, xTest, yTest, id) {
    //async function trainModel(xTrain, yTrain, xTest, yTest, epochFromUi, learningRate, id) {


    let startTime = Date.now();
    let trainingProcess = 0;
    if(fs.existsSync(FILE_SAVE_PATH +'/'+ id +'/model.json')){
        console.log('model exist... load model...');

        /**
         * load model
         */
        const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH +'/' + id +'/model.json'}`);
        const optimizer = tf.train.adam(learningRate);

        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        const history = await model.fit(xTrain, yTrain, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [xTest, yTest],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    // console.log('epoch: '+epoch);
                    await tf.nextFrame();
                    if(epoch % 10 === 0){
                        trainingProcess = epoch / epochs * 100;
                        createOrUpdateTraningProcess(id, trainingProcess);
                    }
                },
                onTrainEnd: async (epoch, logs) => {
                    nnStatus.setStatusToFinished(id);
                    await model.save(`file://${FILE_SAVE_PATH}/` + id);
                    console.log('model saved!')
                    createOrUpdateTraningProcess(id, 100);
                },
                onEpochBegin: async (epoch) => {
                    //console.log('epoch: '+ (epoch+1) +'/'+epochs+' started.')
                },
                onTrainBegin: async () => {
                    createOrUpdateTraningProcess(id, 0);
                }
            }
        });
        const evalOutput = model.evaluate(xTest, yTest);

        let predictionInPercent = (evalOutput[1].dataSync()[0].toFixed(4)) * 100;
        let loss = evalOutput[0].dataSync()[0].toFixed(10);

        let endTime = Date.now();
        let TRAINING_DURATION = endTime-startTime;
        TRAINING_DURATION = (TRAINING_DURATION/1000/60).toFixed(1);

        createStatistics(TRAINING_DURATION, predictionInPercent.toFixed(), loss, epochs, batchSize, id);


    }else{
        let useDefaultModel = false;
        const model = tf.sequential();

        /**
         * Input Layer
         */
        /*
        model.add(tf.layers.dense({inputShape: [xTrain.shape[1]],
            units: 50,
            useBias: true,
            activation: activationFunction !== null ? activationFunction :'relu',
        }));
        */

/*
        hiddenNeurons.forEach(function (value, index, array) {

            if(value[0] === '' || typeof value[0] === 'undefined' || value[0] === null ||
                value[1] === '' || typeof value[1] === 'undefined' || value[1] === null) {
                useDefaultModel = true;
            }

            if(!useDefaultModel){

                 if(value[0].toLowerCase().includes('dense')){
                    model.add(tf.layers.dense({
                        units: value[1],
                        useBias: true,
                        activation: activationFunction !== null ? activationFunction :'relu'
                    }));
                }
            }
        });
        */

        if(!useDefaultModel){
            /**
             * Output Layer
             */
           // model.add(tf.layers.dense({units: 3, useBias: true, activation: 'softmax'}));

            model.add(tf.layers.dense({inputShape: [xTrain.shape[1]], units: 50, useBias: true, activation: 'relu'}));
            model.add(tf.layers.dense({units: 30, useBias: true, activation: 'tanh'}));
            model.add(tf.layers.dense({units: 20, useBias: true, activation: 'relu'}));
            model.add(tf.layers.dense({units: 3, useBias: true, activation: 'softmax'}));

            model.summary();

            const optimizer = tf.train.adam(learningRate);

            model.compile({
                optimizer: optimizer,
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy'],
            });

            const history = await model.fit(xTrain, yTrain, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [xTest, yTest],
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // console.log('epoch: '+epoch);
                        await tf.nextFrame();
                        if(epoch % 10 === 0){
                            trainingProcess = epoch / epochs * 100;
                            createOrUpdateTraningProcess(id, trainingProcess);
                        }
                    },
                    onTrainEnd: async (epoch, logs) => {
                        nnStatus.setStatusToFinished(id);
                        await model.save(`file://${FILE_SAVE_PATH}/` + id);
                        console.log('model saved!')
                        createOrUpdateTraningProcess(id, 100);
                    },
                    onEpochBegin: async (epoch) => {
                       // console.log('epoch: '+ (epoch+1) +'/'+epochs+' started.')
                    },
                    onTrainBegin: async () => {
                        createOrUpdateTraningProcess(id, 0);
                    }
                }
            });

        } else {

            /**
             * Default model
             */
            tf.layers.batchNormalization();
            model.add(tf.layers.dense({
                    inputShape: [xTrain.shape[1]],
                    activation: 'relu',
                    units: 10
                }
            ));
            model.add(tf.layers.dense({
                    activation: 'relu',
                    units: 8
                }
            ));
            model.add(tf.layers.dense({
                    activation: 'relu',
                    units: 6
                }
            ));
            model.add(tf.layers.dense({
                    activation: 'relu',
                    units: 4
                }
            ));
            model.add(tf.layers.dense({
                    activation: 'relu',
                    units: 2
                }
            ));

            model.add(tf.layers.dense({
                    units: 3,
                    activation: 'softmax'
                }
            ));

            model.summary();

            const optimizer = tf.train.adam(learningRate);

            model.compile({
                optimizer: optimizer,
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy'],
            });

            const history = await model.fit(xTrain, yTrain, {
                epochs: epochs,
                batchSize: batchSize,
                validationData: [xTest, yTest],
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // console.log('epoch: '+epoch);
                        await tf.nextFrame();
                        if(epoch % 10 === 0){
                            trainingProcess = epoch / epochs * 100;
                            createOrUpdateTraningProcess(id, trainingProcess);
                        }
                    },
                    onTrainEnd: async (epoch, logs) => {
                        nnStatus.setStatusToFinished(id);
                        await model.save(`file://${FILE_SAVE_PATH}/` + id);
                        console.log('model saved!')
                        createOrUpdateTraningProcess(id, 100);
                    },
                    onEpochBegin: async (epoch) => {
                       // console.log('epoch: '+ (epoch+1) +'/'+epochs+' started.')
                    },
                    onTrainBegin: async () => {
                        createOrUpdateTraningProcess(id, 0);
                    }
                }
            });
        }



        const evalOutput = model.evaluate(xTest, yTest);

        let predictionInPercent = (evalOutput[1].dataSync()[0].toFixed(4)) * 100;
        let loss = evalOutput[0].dataSync()[0].toFixed(10);

        let endTime = Date.now();
        let TRAINING_DURATION = endTime-startTime;
        TRAINING_DURATION = (TRAINING_DURATION/1000/60).toFixed(1);


        createStatistics(TRAINING_DURATION, predictionInPercent, loss, epochs, batchSize, id);

    }

};


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

module.exports = {

    wineTrainer: async function (id) {


        console.log('converter.getWineData() start');
        const [xTrain, yTrain, xTest, yTest] = await converter.getWineData();
        console.log('converter.getWineData() end');

        request(addresses.getVinnslServiceEndpoint() + '/vinnsl/' + id,  async function (error, response, body) {
            try {
                let json = convert.xml2json(body, options);
                let vinnslJSON = JSON.parse(json);
                inputNeurons = getinputNeurons(vinnslJSON);
                outputNeurons = getOutputNeurons(vinnslJSON);
                activationFunction = getActivationFunction(vinnslJSON);
                epochs = getIterations(vinnslJSON);
                learningRate = getLearningRate(vinnslJSON);
                batchSize = getBatchSize(vinnslJSON);
                getHiddenCount(vinnslJSON);
                labelIndex = getLabelIndex(vinnslJSON);
                schemaID = getDataSchemaID(vinnslJSON);

                console.log('trainModel start');
                let model = await trainModel(xTrain, yTrain, xTest, yTest, id);
                console.log('trainModel end');

            }catch (e) {
                console.log(e);
            }
        });

    }
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
                            if(vinnslJSON.vinnsl.definition.parameters.valueparameter[i]._attributes.name === 'iterations'){
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
                       // for(i in vinnslJSON.vinnsl.definition.parameters.comboparameter){
                            if(vinnslJSON.vinnsl.definition.parameters.comboparameter._attributes.name === 'activationfunction'){
                                return vinnslJSON.vinnsl.definition.parameters.comboparameter._text;
                            }
                       // }
                    }
                }
            }
        }
    }
    return DEFAULT_ACTIVATION_FUNCTON;
}


/**
 * Get the name of the file e.g. (iris.txt)
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
