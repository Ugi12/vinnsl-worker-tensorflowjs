
const requestPromise = require('request-promise');
const tf = require('@tensorflow/tfjs-node');
const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/lstm';
const helper = require('../../util/lstm/helper');
const fs = require('fs');
const addresses = require('../../util/addresses');
const request = require('request-promise');
const TEXT_PATH = 'public/nnworker/data/lstm';

const _ = require('lodash');
//import _ from 'lodash'



function createWordMap(textData) {
    const wordArray = textData.split(' ')
    const countedWordObject = wordArray.reduce((acc, cur, i) => {
        if (acc[cur] === undefined) {
            acc[cur] = 1
        } else {
            acc[cur] += 1
        }
        return acc
    }, {})

    const arraOfshit = []
    for (let key in countedWordObject) {
        arraOfshit.push({ word: key, occurence: countedWordObject[key] })
    }

    const wordMap = _.sortBy(arraOfshit, 'occurence').reverse().map((e, i) => {
        e['code'] = i
        return e
    })

    return wordMap
}

function getSamples(){
    const startOfSeq = _.random(0, endOfSeq, false)
    const retVal = preparedDataforTestSet.slice(startOfSeq, startOfSeq + (examinedNumberOfWord + 1))
    return retVal
}

function toSymbol(word){
    const object = wordMap.filter(e => e.word === word)[0]
    return object.code
}
function decode(probDistVector) {
    // It could be swithced to tf.argMax(), but I experiment with values below treshold.
    const probs = probDistVector.softmax().dataSync()
    const maxOfProbs = _.max(probs)
    const probIndexes = []

    for (let prob of probs) {
        if (prob > (maxOfProbs - 0.3)) {
            probIndexes.push(probs.indexOf(prob))
        }
    }

    return probIndexes[_.random(0, probIndexes.length - 1)]
}
// return a word
function fromSymbol(symbol){
    const object = wordMap.filter(e => e.code === symbol)[0]
    return object.word
}

function loadTrainingText(id) {
    return requestPromise(addresses.getVinnslServiceEndpoint() + '/vinnsl/get-text/lstm/' + id)
}


////VALUES /////
let wordMap = [];
let wordMapLength = null;
const learning_rate = 0.001;
const examinedNumberOfWord = 6;
let inputText = '';
let preparedDataforTestSet = '';
let endOfSeq = null;

module.exports = {

    lstmTextGeneration: async function (id, generateLength, temparature) {


        /** FUNKTIONIERT
        if(fs.existsSync(FILE_SAVE_PATH +'/'+ id +'/model.json')){

            inputText = await loadTrainingText(id);

            preparedDataforTestSet = inputText.split(' ')

            endOfSeq = preparedDataforTestSet.length - (examinedNumberOfWord + 1)
            const optimizer = tf.train.rmsprop(learning_rate)


            wordMap =  await createWordMap(inputText);
            wordMapLength = Object.keys(wordMap).length;

            console.log('model exist... load model...');

            const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH + '/'+ id +'/model.json'}`);


            const symbolCollector = _.shuffle(getSamples()).map(s => {
                return toSymbol(s)
            })



            let sampleLen = 40;
            let charSet = await helper.getChartSet(inputText);
            [seedSentence, seedSentenceIndices] = await helper.getRandomSlice(inputText, sampleLen, charSet);


            for (let i = 0; i < 30; i++) {
                const inputBuffer = new tf.TensorBuffer([1, sampleLen, charSet.length]);
                inputBuffer.set(1, 0, i, seedSentenceIndices[i]);
                const input = inputBuffer.toTensor();
                const output = model.predict(input);
                symbolCollector.push(decode(output));
            }


            const generatedText = symbolCollector.map(s => {
                return fromSymbol(s)
            }).join(' ')

            console.log(generatedText);
            return generatedText;
        }else {
            console.log("model not exist... please train your network first.")
        }
        */


        /**
         * 9.7.2019
         *
        if(fs.existsSync(FILE_SAVE_PATH +'/'+ id +'/model.json')){

            console.log('model exist... load model...');

            const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH + '/'+ id +'/model.json'}`);

            inputText = await loadTrainingText(id);
            const sampleLen = model.inputs[0].shape[1];
            const charSetSize = model.inputs[0].shape[2];

            const symbolCollector = _.shuffle(getSamples()).map(s => {
                return toSymbol(s)
            })



            //let sampleLen = 40;
            let charSet = await helper.getChartSet(inputText);
            [seedSentence, seedSentenceIndices] = await helper.getRandomSlice(inputText, sampleLen, charSet);

            let length = 200;
            let temperature = 0.75;
            let generated = '';
            while (generated.length < length) {
                let inputBuffer = new tf.TensorBuffer([1, sampleLen, charSet.length]);

                for (let i = 0; i < sampleLen; i++) {
                    inputBuffer.set(1, 0, i, seedSentenceIndices[i]);
                }


                const input = inputBuffer.toTensor();
                const output = model.predict(input);

                const winnerIndex = sample(tf.squeeze(output), temperature);
                const winnerChar = charSet[winnerIndex];

                generated += winnerChar;
                seedSentenceIndices = seedSentenceIndices.slice(1);
                seedSentenceIndices.push(winnerIndex);
                input.dispose();
                output.dispose();

            }





            console.log(generated);
            return generated;
        }
         */




        if (fs.existsSync(FILE_SAVE_PATH + '/' + id + '/model.json')) {

            console.log('model exist... load model...');

            let seedSentence;
            let seedSentenceIndices;

            const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH + '/' + id + '/model.json'}`);

            inputText = await loadTrainingText(id);
            const sampleLen = model.inputs[0].shape[1];
            const charSetSize = model.inputs[0].shape[2];

            let charSet = await helper.getChartSet(inputText);

            //get seed sentence from the data
            [seedSentence, seedSentenceIndices] = await helper.getRandomSlice(inputText, sampleLen, charSet);


            seedSentenceIndices = seedSentenceIndices.slice();

            let length = 200;
            let temperature = 0.75;
            let generated = '';
            while (generated.length < length) {
                let inputBuffer = new tf.TensorBuffer([1, sampleLen, charSetSize]);

                for (let i = 0; i < sampleLen; i++) {
                    inputBuffer.set(1, 0, i, seedSentenceIndices[i]);
                }


                const input = inputBuffer.toTensor();
                const output = model.predict(input);

                const winnerIndex = sample(tf.squeeze(output), temperature);
                const winnerChar = charSet[winnerIndex];

                //TODO:
                //await tf.nextFrame();

                generated += winnerChar;
                seedSentenceIndices = seedSentenceIndices.slice(1);
                seedSentenceIndices.push(winnerIndex);

                input.dispose();
                output.dispose();

            }

            console.log(generated);
            return generated;
        } else {
            console.log("model not exist... please train your network first.")
        }


    }

};

function sample(probs, temperature) {
    return tf.tidy(() => {
        const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
        const isNormalized = false;
        // `logits` is for a multinomial distribution, scaled by the temperature.
        // We randomly draw a sample from the distribution.
        return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
    });
}


const train = async (numIterations) => {
    let lossCounter = null

    for (let iter = 0; iter < numIterations; iter++) {

        let labelProbVector
        let lossValue
        let pred
        let losse
        let samplesTensor

        const samples = getSamples().map(s => {
            return toSymbol(s)
        })

        labelProbVector = encode(samples.splice(-1))

        if (stop_training) {
            stop_training = false
            break
        }

        // optimizer.minimize is where the training happens.

        // The function it takes must return a numerical estimate (i.e. loss)
        // of how well we are doing using the current state of
        // the variables we created at the start.

        // This optimizer does the 'backward' step of our training process
        // updating variables defined previously in order to minimize the
        // loss.
        lossValue = optimizer.minimize(() => {
            // Feed the examples into the model
            samplesTensor = tf.tensor(samples, [1, examinedNumberOfWord, 1])
            pred = predict(samplesTensor);
            losse = loss(labelProbVector, pred);
            return losse
        }, true);

        if (lossCounter === null) {
            lossCounter = lossValue.dataSync()[0]
        }
        lossCounter += lossValue.dataSync()[0]


        if (iter % 100 === 0 && iter > 50) {
            const lvdsy = lossCounter / 100
            lossCounter = 0
            console.log(`
            --------
            Step number: ${iter}
            The average loss is (last 100 steps):  ${lvdsy}
            Number of tensors in memory: ${tf.memory().numTensors}
            --------`)
            chart.data.datasets[0].data.push(lvdsy)
            chart.data.labels.push(iter)
            chart.update()
        }

        // Use tf.nextFrame to not block the browser.
        await tf.nextFrame();
        pred.dispose()
        labelProbVector.dispose()
        lossValue.dispose()
        losse.dispose()
        samplesTensor.dispose()
    }
}
async function loadTrainingTest(id) {
    return request(addresses.getVinnslServiceEndpoint() + '/vinnsl/get-text/lstm/' + id)
};