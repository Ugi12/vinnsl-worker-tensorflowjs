//  const axios = require('axios')

const requestPromise = require('request-promise');
const request = require('request');
const fs = require('fs');
const convert = require('xml-js');
const tf = require('@tensorflow/tfjs-node');
const dateFormat = require('dateformat');
const addresses = require('../../util/addresses');
const nnStatus = require('../../util/nn-status')

const helper = require('../../util/lstm/helper');

const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/lstm';
const TEXT_PATH = 'public/nnworker/data/lstm';


const INPUT_NEURONS = 100;
const DEFAULT_LEARNING_RATE = 0.01;
const DEFAULT_EPOCHS = 20;
const DEFAULT_BATCH_SIZE = 128;
const DEFAULT_ACTIVATION_FUNCTON = 'softmax'

let TRAINING_DURATION = 0;

//data from xml
let inputNeurons = null;
let hiddenlayers = null;
let hiddenNeurons = [];
let activationFunction = null;

//
// let epochs = 1;
let examplesPerEpoch = 2048;
//let batchSize = null;
let lstmLayerSizes = [];
let validationSplit = 0.0625;
let learningRate = null;
let sampleLen = 40;
let sampleStep = 3;
let charSet = [];
let indicesUint16Array = [];
let examplePosition = 0;
let shuffledBeginIndices = [];
let text = '';


////////////
let seqLength = 5;
let outputKeepProb = 0.2;
let epochs = 100;
let batchSize = 200;

const LSTM_LAYERS = 2;
const LSTM_SIZE = 128;
//////////

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



function loadTrainingText(id) {
    return requestPromise(addresses.getVinnslServiceEndpoint() + '/vinnsl/get-text/lstm/' + id)
}
function loadVinnslXML(id){
    return requestPromise(addresses.getVinnslServiceEndpoint() + '/vinnsl/' + id);
}

module.exports = {

    lstmTrainer: async function (id, t) {



        try{

            text = await loadTrainingText(id);
            //text =  await getText();
            const VINNSL = await loadVinnslXML(id);

            let json = convert.xml2json(VINNSL, options);
            let vinnslJSON = JSON.parse(json);

            activationFunction = getActivationFunction(vinnslJSON);
            //
            // epochs = getEpochs(vinnslJSON);
            batchSize = getBatchSize(vinnslJSON);
            learningRate = getLearningRate(vinnslJSON);

            inputNeurons = getinputNeurons(vinnslJSON);
            lstmLayerSizes.push(inputNeurons);
            getHiddenCount(vinnslJSON);
            for(let i = 0; i < hiddenNeurons.length; ++i){
                lstmLayerSizes.push(hiddenNeurons[i]);
            }



            charSet = await helper.getChartSet(text);
            indicesUint16Array = await  helper.converTextToIndices(text, charSet);
            shuffledBeginIndices = await helper.generateBeginIndices(text.length, sampleLen, sampleStep);





            let startTime = Date.now();
            if(fs.existsSync(FILE_SAVE_PATH +'/'+ id +'/model.json')){

                let startTime = Date.now();

                console.log('model exist... load model...');

                const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH + '/'+ id +'/model.json'}`);

                const optimizer = tf.train.rmsprop(0.001);
                model.compile({
                    optimizer: optimizer,
                    loss: 'categoricalCrossentropy',
                    metrics: ['accuracy']
                })

                model.summary();

                let batchCount = 0;
                const batchesPerEpoch = examplesPerEpoch / batchSize;
                const totalBatches = epochs * (batchesPerEpoch-1);

                await tf.nextFrame();

                let trainingProcess = 0;
                createOrUpdateTraningProcess(id, 0);

                let loss = 0;
                let predInpercent = 0;
                for(let i = 0 ;i < epochs ; ++i){

                    const [xs, ys] = nextDataEpoch();
                    await model.fit(xs, ys, {
                        epochs: 1,
                        batchSize: batchSize,
                        validationSplit: validationSplit,
                        callbacks: {
                            onBatchEnd: async (batch, logs) => {
                                console.log("batchEnd: " + ++batchCount + "/" + totalBatches);
                                if(batchCount % 10 == 0){
                                    trainingProcess = (batchCount / totalBatches *100).toFixed(1);
                                    createOrUpdateTraningProcess(id, trainingProcess);
                                }
                            },
                            onTrainEnd: async (batch, logs) => {
                                const evalOutput = model.evaluate(xs, ys);
                                predInpercent = (evalOutput[1].dataSync()[0].toFixed(4)) * 100;
                                loss = evalOutput[0].dataSync()[0].toFixed(3);

                            }
                        }
                    })
                    xs.dispose();
                    ys.dispose();
                }
                createOrUpdateTraningProcess(id, 100);

                 let endTime = Date.now();
                TRAINING_DURATION = endTime-startTime;
                TRAINING_DURATION = (TRAINING_DURATION/1000/60).toFixed(0);
                createStatistics(TRAINING_DURATION, predInpercent, loss, epochs, batchSize, id);

                nnStatus.setStatusToFinished(id);
                await model.save(`file://${FILE_SAVE_PATH}/` + id);
                console.log('model saved. ');


            }else{
                let startTime = Date.now();

                console.log("create model");
                const model =  await createModel();

                let batchCount = 0;
                const batchesPerEpoch = examplesPerEpoch / batchSize;
                const totalBatches = epochs * (batchesPerEpoch-1);

                await tf.nextFrame();

                let trainingProcess = 0;
                createOrUpdateTraningProcess(id, 0);
                let loss = 0;
                let predInpercent = 0;
                for(let i = 0 ;i < epochs ; ++i){

                    const [xs, ys] = nextDataEpoch();
                    await model.fit(xs, ys, {
                        epochs: 1,
                        batchSize: batchSize,
                        validationSplit: validationSplit,
                        callbacks: {
                            onBatchEnd: async (batch, logs) => {
                                console.log("batchEnd: " + ++batchCount + "/" + totalBatches);
                                if(batchCount % 10 == 0){
                                    trainingProcess = (batchCount / totalBatches *100).toFixed(1);
                                    createOrUpdateTraningProcess(id, trainingProcess);
                                }
                            },
                            onTrainEnd: async (batch, logs) => {
                                const evalOutput = model.evaluate(xs, ys);
                                predInpercent = (evalOutput[1].dataSync()[0].toFixed(4)) * 100;
                                loss = evalOutput[0].dataSync()[0].toFixed(3);

                            }
                        }
                    })
                    xs.dispose();
                    ys.dispose();
                }


                createOrUpdateTraningProcess(id, 100);

                createStatistics(TRAINING_DURATION, predInpercent, loss, epochs, batchSize, id);

                nnStatus.setStatusToFinished(id);
                await model.save(`file://${FILE_SAVE_PATH}/` + id);
                console.log('model saved. ');

            }


            //});


        }catch (e) {
            console.log(e);
        }
    },
    deleteLstmModel: function (id) {
        var path = FILE_SAVE_PATH +'/'+ id;
        if(fs.existsSync(path)){

            fs.readdirSync(path).forEach(function(file,index){
                fs.unlinkSync(path + "/" + file);
            });
            fs.rmdirSync(FILE_SAVE_PATH +'/'+ id);
            console.log('model: '+ id +' deleted');
        }
    }
}

function createOrUpdateTraningProcess(id, trainingInPercent) {

    /*
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
    */

    request.post(addresses.getVinnslServiceEndpoint() + '/vinnsl/create-update/process', {
        json: {
            id: id,
            trainingProcess: trainingInPercent
        }
    }, (error, res, body) => {
        if (error) {
            console.error(error)
            return
        }
        // console.log(`statusCode: ${res.statusCode}`)
        // console.log(body)
    });

}


function createModel(){

    const model = tf.sequential();

    for(let i = 0 ; i < lstmLayerSizes.length; ++i){
        const lstmLayerSize = lstmLayerSizes[i];
        model.add(tf.layers.lstm({
            units: lstmLayerSize,
            returnSequences: i < lstmLayerSizes.length - 1,
            inputShape: i === 0 ? [sampleLen, charSet.length] : undefined
        }))
    }

    model.add(tf.layers.dense({units: charSet.length, activation: 'softmax'}));

    const optimizer = tf.train.rmsprop(learningRate);
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    model.summary();



/*
    console.log("setting up model...");

    let cells = [];
    for(let i = 0; i < LSTM_LAYERS; i++) {
        const cell = tf.layers.lstmCell({
            units: LSTM_SIZE
        });
        cells.push(cell);
    }

    const multiLstmCellLayer = tf.layers.rnn({
        cell: cells,
        returnSequences: true,
        inputShape: [sampleLen, charSet.length]
    });

    const dropoutLayer = tf.layers.dropout({
        rate: 0.2
    });

    const flattenLayer = tf.layers.flatten();

    const denseLayer = tf.layers.dense({
        units: charSet.length,
        activation: 'softmax',
        useBias: true
    });

    const model = tf.sequential();
    model.add(multiLstmCellLayer);
    model.add(dropoutLayer);
    model.add(flattenLayer);
    model.add(denseLayer);

    model.summary();

    console.log("compiling...");

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: 'adam'
    });

    console.log("done.");
*/
    return model;
}


function nextDataEpoch(){
    const xsBuffer = new tf.TensorBuffer([examplesPerEpoch, sampleLen, charSet.length]);
    const ysBuffer = new tf.TensorBuffer([examplesPerEpoch, charSet.length]);
    for(let i = 0; i < examplesPerEpoch; ++i){
        const beginIndex = shuffledBeginIndices[examplePosition % shuffledBeginIndices.length];
        for(let j = 0; j < sampleLen; ++j){
            xsBuffer.set(1, i, j, indicesUint16Array[beginIndex + j]);
        }
        ysBuffer.set(1, i, indicesUint16Array[beginIndex + sampleLen]);
        examplePosition++;
    }
    return [xsBuffer.toTensor(), ysBuffer.toTensor()];
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
function getEpochs(vinnslJSON) {
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
                        // for(i in vinnslJSON.vinnsl.definition.parameters.comboparameter){
                        if(vinnslJSON.vinnsl.definition.parameters.comboparameter._attributes.name === 'activationfunction'){
                            return vinnslJSON.vinnsl.definition.parameters.comboparameter._text;
                        }
                        //}
                    }
                }
            }
        }
    }
    return DEFAULT_ACTIVATION_FUNCTON;
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
 * get the number of hidden layer/neurons
 * @param vinnslJSON
 * @returns {number} of hidden layer/neurons
 */
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
                                hiddenNeurons.push(parseInt(vinnslJSON.vinnsl.definition.structure.hidden.size._text));

                            }else if(Array.isArray(obj) && obj.length > 0){
                                hiddenlayers = vinnslJSON.vinnsl.definition.structure.hidden.length;
                                for(i in vinnslJSON.vinnsl.definition.structure.hidden){
                                    hiddenNeurons.push(parseInt(vinnslJSON.vinnsl.definition.structure.hidden[i].size._text,10));
                                }
                            }

                        }
                    }
                }
            }
        }
    }
}

function createStatistics(trainingTime, bestResult, loss, epochs, batchSize, id) {

    /*
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

        */

    request.post(addresses.getVinnslServiceEndpoint() + '/vinnsl/create-update-js/statistic', {
        json: {
            id: id,
            createTimestamp: dateFormat(new Date(), "UTC:dd.mm.yyyy hh:MM:ss TT"),
            trainingTime: trainingTime ,
            numberOfTraining: 1,
            lastResult: bestResult,
            bestResult: bestResult,
            epochs: epochs,
            loss: loss,
            batchSize: batchSize
        }
    }, (error, res, body) => {
        if (error) {
            console.error(error)
            return
        }
        //console.log(`statusCode: ${res.statusCode}`)
        //console.log(body)
    });
}


async function getText() {
    return `long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .`;

}











