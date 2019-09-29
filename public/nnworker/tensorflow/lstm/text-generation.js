

const tf = require('@tensorflow/tfjs');
const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/lstm';
const helper = require('../../util/lstm/helper');
const fs = require('fs');
const addresses = require('../../util/addresses');
const request = require('request-promise');
const TEXT_PATH = 'public/nnworker/data/lstm';

const _ = require('lodash');
//import _ from 'lodash'



const inputText = `long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .`


const numIterations = 20000
const learning_rate = 0.001
const rnn_hidden = 64
const preparedDataforTestSet = inputText.split(' ')
const examinedNumberOfWord = 6
const endOfSeq = preparedDataforTestSet.length - (examinedNumberOfWord + 1)
const optimizer = tf.train.rmsprop(learning_rate)
let chart
let stop_training = false

const createWordMap = (textData) => {
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


const wordMap = createWordMap(inputText);
const wordMapLength = Object.keys(wordMap).length;



const getSamples = () => {
    const startOfSeq = _.random(0, endOfSeq, false)
    const retVal = preparedDataforTestSet.slice(startOfSeq, startOfSeq + (examinedNumberOfWord + 1))
    return retVal
}

const toSymbol = (word) => {
    const object = wordMap.filter(e => e.word === word)[0]
    return object.code
}
const decode = (probDistVector) => {
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
const fromSymbol = (symbol) => {
    const object = wordMap.filter(e => e.code === symbol)[0]
    return object.word
}

module.exports = {
/**
    lstmTextGeneration: async function (id, generateLength, temparature) {

        //const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH + '/' + id + '/model.json'}`);
        await train(5000);

        const symbolCollector = _.shuffle(getSamples()).map(s => {
            return toSymbol(s)
        })

        for (let i = 0; i < 30; i++) {
            const predProbVector = model.predict(tf.tensor(symbolCollector.slice(-examinedNumberOfWord), [1, examinedNumberOfWord,1]))
            symbolCollector.push(decode(predProbVector));
        }

        const generatedText = symbolCollector.map(s => {
            return fromSymbol(s)
        }).join(' ')
        console.log(generatedText)
    }
    */

    lstmTextGeneration: async function (id, generateLength, temparature) {

        const DEFAULT_TEXT_LEN = 200;
        const DEFAULT_TEMPERATURE = 0.25;
        let sampleLen = 40;
        let seedSentence;
        let seedSentenceIndices;
        let charSet = [];

        if(fs.existsSync(FILE_SAVE_PATH +'/'+ id +'/model.json')){

            //await request(addresses.getVinnslServiceEndpoint() + '/vinnsl/get-text/lstm/' + id,  async function (error, response, body) {

            //text = response.body;
            //text = fs.readFileSync(TEXT_PATH +'/generated.txt', 'utf8');

            if(!(generateLength > 0) || (temparature <=0 || temparature > 1)){
                generateLength = DEFAULT_TEXT_LEN;
                temparature = DEFAULT_TEMPERATURE;
            }

            const text = await loadTrainingTest(id);

            charSet = await helper.getChartSet(text);
            [seedSentence, seedSentenceIndices] = await helper.getRandomSlice(text, sampleLen, charSet);
            const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH + '/' + id + '/model.json'}`);


            const temperatureScalar = tf.scalar(temparature);

            //sampleLen = model.inputs[0].shape[1];
            //const charSetSize = model.inputs[0].shape[2];
            const charSetSize = charSet.length;

            seedSentenceIndices = seedSentenceIndices.slice();

            let generatedText = '';
            while (generatedText.length < generateLength) {


                const inputBuffer = new tf.TensorBuffer([1, sampleLen, charSetSize]);
                for (let i = 0; i < sampleLen; ++i) {
                    inputBuffer.set(1, 0, i, seedSentenceIndices[i]);
                }
                const input = inputBuffer.toTensor();
                const output = model.predict(input);

                const winnerIndex = await helper.sample(tf.squeeze(output), temperatureScalar);
                const winnerChar = charSet[winnerIndex];
                generatedText += winnerChar;
                //console.log('generatedText'+generatedText);

                seedSentenceIndices = seedSentenceIndices.slice(1);
                seedSentenceIndices.push(winnerChar);

                input.dispose();
                output.dispose();
            }
            temperatureScalar.dispose();
            console.log(generatedText);
            return generatedText;
            // });
        }

    }

};

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