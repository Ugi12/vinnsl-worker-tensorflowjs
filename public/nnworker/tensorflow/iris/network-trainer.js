const axios = require('axios')

let request = require('request');
const fs = require('fs');
let convert = require('xml-js');
const tf = require('@tensorflow/tfjs-node');
let dateFormat = require('dateformat');

const converter = require('../../util/iris/converter')
const irisJS = require('../../data/iris/iris');
const table = require('../../util/iris/table');
let addresses = require('../../util/addresses');
const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/iris';

//default values
const INPUT_NEURONS = 4;
const OUTPUT_NEURONS = 3;
const LABEL_INDEX = 0;
const DEFAULT_LEARNING_RATE = 0.01;
const DEFAULT_EPOCHS = 100;

//data from xml
let inputNeurons = null;
let outputNeurons = null;
let hiddenlayers = null;
let hiddenNeurons = [];
let activationFunction = null;
let epochsGlobal = null;

let TRAINING_DURATION = 0;

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
module.exports = {

    irisTrainer: async function (id, vinnslItem,  epochs, learningrate, res) {

        request(addresses.getVinnslServiceEndpoint() + '/vinnsl/' + id,  function (error, response, body) {
            try{
                let json = convert.xml2json(body, options);
                let vinnslJSON = JSON.parse(json);
                inputNeurons = getinputNeurons(vinnslJSON);
                outputNeurons = getOutputNeurons(vinnslJSON);
                activationFunction = getActivationFunction(vinnslJSON);
                epochs = getIterations(vinnslJSON);
                epochsGlobal = epochs;
                //learningRate = getLearningRate(vinnslJSON);
                getHiddenCount(vinnslJSON);
                labelIndex = getLabelIndex(vinnslJSON);
                schemaID = getDataSchemaID(vinnslJSON);

                learningRate = getLearningRate(vinnslJSON);

                }catch (e) {
                    console.log(e);
                }
        });


        /**
         * get iris data
         */
         request(addresses.getVinnslStorageServiceEndpoint() + '/files/' + schemaID, async function (error, response, body) {


             if (!error && response.statusCode == 200) {

                 const convertedIrisData = converter.csvToArray(body);
                 const [xTrain, yTrain, xTest, yTest] = converter.getIrisData(convertedIrisData, 0.15);

                 let model = await trainModel(xTrain, yTrain, xTest, yTest, id);

                 const xData = xTest.dataSync();
                 const yTrue = yTest.argMax(-1).dataSync();

                 const predictOut = model.predict(xTest);
                 const yPred = predictOut.argMax(-1);

                 //const logits = Array.from(predictOut.dataSync());
                 //const winner = irisJS.getIrisClasses([predictOut.argMax(-1).dataSync()[0]]);

                 let predictionInPercent = await table.renderResultTable(xData, yTrue, yPred.dataSync(), predictOut.dataSync(), res, id);

                 createStatistics(TRAINING_DURATION, predictionInPercent, epochs, learningRate, id);

             } else {

                 //on request fail. get default iris data and train model
                 const [xTrain, yTrain, xTest, yTest] = converter.getIrisData([], 0.15);

                 let model = await trainModel(xTrain, yTrain, xTest, yTest, id);

                 const xData = xTest.dataSync();
                 const yTrue = yTest.argMax(-1).dataSync();

                 const predictOut = model.predict(xTest);
                 const yPred = predictOut.argMax(-1);

                 //const logits = Array.from(predictOut.dataSync());
                 //const winner = irisJS.getIrisClasses([predictOut.argMax(-1).dataSync()[0]]);

                 let predictionInPercent = await table.renderResultTable(xData, yTrue, yPred.dataSync(), predictOut.dataSync(), res, id);

                 createStatistics(TRAINING_DURATION, predictionInPercent, epochs, learningRate, id);

             }

         });

    },function(err){
        console.log(err);
    }
    
};


async function trainModel(xTrain, yTrain, xTest, yTest, id) {

    try {
        /**
         * check if exist pre trained model
         */
        if(fs.existsSync(FILE_SAVE_PATH +'/'+ id +'/model.json')){


            console.log('file exist.. load and train model');
            const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH +'/' + id +'/model.json'}`);

            const optimizer = tf.train.adam(learningRate);

            model.compile({
                optimizer: optimizer,
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy'],
            });

            let startTime = Date.now();
            let epochCounter = 0;
            const history = await model.fit(xTrain, yTrain, {
                epochs: epochsGlobal,
                validationData: [xTest, yTest],
                callbacks: {
                    onTrainBegin: async () => {
                        createOrUpdateTraningProcess(id, 0);
                    },
                    onEpochEnd: async () => {
                        let trainingProcess = ++epochCounter / epochsGlobal * 100;
                        createOrUpdateTraningProcess(id, trainingProcess);
                    },
                    onTrainEnd: async (epoch, logs) => {
                        await model.save(`file://${FILE_SAVE_PATH}/`+id );
                        createOrUpdateTraningProcess(id, 100);
                        console.log('model saved!')
                    }
                }
            });

            let endTime = Date.now();
            TRAINING_DURATION = endTime-startTime;
            TRAINING_DURATION = (TRAINING_DURATION/1000).toFixed(1);
            return model;

        }else{
            fs.mkdirSync(FILE_SAVE_PATH+'/'+id)
            console.log('model not exist.. create and train model..');
            const model = tf.sequential();
/*
            model.add(tf.layers.dense({
                    inputShape: [xTrain.shape[1]],
                    activation: 'sigmoid',
                    units: 10
                }
            ));
*/
            /**
             * Input Layer
             */
            model.add(tf.layers.dense({
                inputShape: [xTrain.shape[1]],
                activation: activationFunction !== null ? activationFunction :'sigmoid',
                units: inputNeurons !== null ? inputNeurons : INPUT_NEURONS
            }));

            /**
             * Hidden Layer/s
             */
            for(let i = 0 ; i < hiddenlayers; i++){
                model.add(tf.layers.dense({
                    units: hiddenNeurons[i],
                    activation: activationFunction !== null ? activationFunction :'sigmoid'
                }))
            }
            /**
             * Output Layer
             */
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

            let startTime = Date.now();
            const history = await model.fit(xTrain, yTrain, {
                epochs: epochsGlobal,
                validationData: [xTest, yTest],
                callbacks: {
                    onTrainBegin: async () => {
                        createOrUpdateTraningProcess(id, 0);
                    },
                    onEpochEnd: async () => {
                        let trainingProcess = ++epochCounter / epochsGlobal * 100;
                        createOrUpdateTraningProcess(id, trainingProcess);
                    },
                    onTrainEnd: async (epoch, logs) => {
                        await model.save(`file://${FILE_SAVE_PATH}/`+id );
                        console.log('model saved!')
                    }
                }
            });

            let endTime = Date.now();
            TRAINING_DURATION = endTime-startTime;
            TRAINING_DURATION = (TRAINING_DURATION/1000).toFixed(1);


            return model;
        }
    }catch (err) {
        console.log(err);
    }

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

function createStatistics(trainingTime, bestResult, epochs, learningRate, id) {

    axios.post(addresses.getVinnslServiceEndpoint() + '/vinnsl/create-update/statistic', {
        id: id,
        createTimestamp: dateFormat(new Date(), "UTC:dd.mm.yyyy hh:MM:ss TT"),
        trainingTime: trainingTime ,
        numberOfTraining: 1,
        lastResult: bestResult,
        bestResult: bestResult,
        epochs: epochs,
        learningRate: learningRate

    })
    .then(function (response) {
        //ignore response
    })
    .catch(function (error) {
        console.log(error);
    });
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
 * @returns name of the activation function
 */
function getActivationFunction(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.definition){
                if(vinnslJSON.vinnsl.definition.parameters){
                    if(vinnslJSON.vinnsl.definition.parameters.comboparameter){
                        //for(i in vinnslJSON.vinnsl.definition.parameters.comboparameter){
                            if(vinnslJSON.vinnsl.definition.parameters.comboparameter._attributes.name === 'activationfunction'){
                                return vinnslJSON.vinnsl.definition.parameters.comboparameter._text;
                            }
                       // }
                    }
                }
            }
        }
    }
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



function convertCSVtoJSON(csv) {


    let headers = ["sepal_length","sepal_width","petal_length","petal_width","species"];

    let lines=csv.split("\n");

    let result = [];

    let invalidDataFound = false;
    //let headers=header.split(",");

    for(let i=0;i<lines.length;i++){

        let obj = {};
        let currentline=lines[i].split(",");

        for(let j=0;j<headers.length;j++){
            if(parseFloat(currentline[j]) >= 0) {
                if(headers[j] === "species"){
                    switch (parseFloat(currentline[j])) {
                        case 0:
                            obj[headers[j]] = "setosa";
                            break;
                        case 1:
                            obj[headers[j]] = "versicolor";
                            break;
                        case 2:
                            obj[headers[j]] = "virginica";
                            break;
                    }
                }else{
                    obj[headers[j]] = parseFloat(currentline[j]);
                }

            }else{
                invalidDataFound = true;
                break;
            }
        }
        if(invalidDataFound){
            invalidDataFound =false;
            continue;
        }
        result.push(obj);

    }

    return result; //JavaScript object
    //return JSON.stringify(result); //JSON

}






