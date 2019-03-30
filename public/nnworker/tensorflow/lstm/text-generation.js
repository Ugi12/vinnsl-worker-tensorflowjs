

const tf = require('@tensorflow/tfjs-node');
const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/lstm';
const helper = require('../../util/lstm/helper');
const fs = require('fs');
const addresses = require('../../util/addresses');
const request = require('request-promise');
const TEXT_PATH = 'public/nnworker/data/lstm';

module.exports = {

    lstmTextGeneration: async function (id, generateLength, temparature) {

        const DEFAULT_TEXT_LEN = 200;
        const DEFAULT_TEMPERATURE = 0.75;
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

            /**
             * Generate text
             */
            const temperatureScalar = tf.scalar(temparature);

            sampleLen = model.inputs[0].shape[1];
            const charSetSize = model.inputs[0].shape[2];

            seedSentenceIndices = seedSentenceIndices.slice();

            let generatedText = '';
            while (generatedText.length < generateLength) {


                const inputBuffer = new tf.TensorBuffer([1, sampleLen, charSetSize]);
                for (let i = 0; i < sampleLen; ++i) {
                    inputBuffer.set(1, 0, i, seedSentenceIndices[i]);
                }
                const input = inputBuffer.toTensor();
                const output = model.predict(input);

                const winnerIndex = await helper.sample(tf.squeeze(output), temparature);
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


async function loadTrainingTest(id) {
    return request(addresses.getVinnslServiceEndpoint() + '/vinnsl/get-text/lstm/' + id)
};