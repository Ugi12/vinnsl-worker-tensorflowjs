const tf = require('@tensorflow/tfjs-node');
const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/mnist';

module.exports = {

    loadMnistModel: async function () {

        const model = await tf.loadLayersModel(`file://${FILE_SAVE_PATH + '/model.json'}`);

        return model;

    }
}