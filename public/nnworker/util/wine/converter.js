const tf = require('@tensorflow/tfjs-node');

const wineJS = require('../../data/wine/wine')

const TEST_DATA_IN_PERCENT = 0.20;





async function toTensors(data, label) {


    //shuffle process
    let shuffleNums = [];
    for(let i = 0; i < data.length; ++i){
        shuffleNums.push(i);
    }
    const shuffledData = [];
    const shuffledLabels = [];

    tf.util.shuffle(shuffleNums);

    for(let i = 0; i < data.length; ++i){
        shuffledData.push(data[shuffleNums[i]]);
        shuffledLabels.push(label[shuffleNums[i]]);
    }

    const amountOfTestData = Math.round(data.length * TEST_DATA_IN_PERCENT);
    const amountOfTrainData  = data.length - amountOfTestData;

    //num of features
    const xDimensions = shuffledData[0].length;

    //convert shuffled data to 2D Tensors
    const xs = tf.tensor2d(shuffledData);
    const ys = tf.oneHot(tf.tensor1d(shuffledLabels).toInt(), wineJS.getNumClasses());


    //split data into train & test sets
    const xTrain = xs.slice([amountOfTestData, 0], [amountOfTrainData, xDimensions]);
    const xTest = xs.slice([0, 0], [amountOfTestData, xDimensions]);
    const yTrain = ys.slice([amountOfTestData, 0], [amountOfTrainData, wineJS.getNumClasses()]);
    const yTest = ys.slice([0, 0], [amountOfTestData, wineJS.getNumClasses()]);

    return [xTrain, yTrain, xTest, yTest];

};

module.exports = {

    getWineData: async function () {

        let data = [];
        let label = [];
        console.log('wineJS.getWineData() start');
        let wineData = await wineJS.getWineData();
        console.log('wineJS.getWineData() end');
        /**
         * split wine data by species
         */
        wineData.forEach(function (element) {
            let templabel = element.slice(0,1);
            let tempdata = element.slice(1,14);
            label.push(templabel[0]-1);
            data.push(tempdata);

        });

        console.log('toTensors() start');
        const [xTrain, yTrain, xTest, yTest] = await toTensors(data, label);
        console.log('toTensors() end');
        return [xTrain, yTrain, xTest, yTest]
    }
}