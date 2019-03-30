const tf = require('@tensorflow/tfjs-node');
const irisJS = require('../../data/iris/iris');




function toTernsors(data, target, testDataInPercent) {

    if(data.length !== target.length){
        throw new Error('invalid data found in converter.toTernsors()');
    }

    //shuffle process
    let shuffleNums = [];
    for(let i = 0; i < data.length; ++i){
        shuffleNums.push(i);
    }
    const shuffledData = [];
    const shuffledTarget = [];

    tf.util.shuffle(shuffleNums);

    for(let i = 0; i < data.length; ++i){
        shuffledData.push(data[shuffleNums[i]]);
        shuffledTarget.push(target[shuffleNums[i]]);
    }

    const amountOfTestData = Math.round(data.length * testDataInPercent);
    const amountOfTrainData  = data.length - amountOfTestData;


    const xDimensions = shuffledData[0].length;
    //const shape = [data.length, xDimensions];

    //convert shuffled data to 2D Tensors
    const xs = tf.tensor2d(shuffledData, [data.length, xDimensions]);

    const ys = tf.oneHot(tf.tensor1d(shuffledTarget).toInt(), irisJS.getIrisNumClasses());

    //split data into training & test.txt sets
    const xTrain = xs.slice([0, 0], [amountOfTrainData, xDimensions]);
    const xTest = xs.slice([amountOfTrainData, 0], [amountOfTestData, xDimensions]);
    const yTrain = ys.slice([0, 0], [amountOfTrainData, irisJS.getIrisNumClasses()]);
    const yTest = ys.slice([0, 0], [amountOfTestData, irisJS.getIrisNumClasses()]);
    return [xTrain, yTrain, xTest, yTest];

}


module.exports = {

    getIrisData: function (convertedIrisData, testDataInPercent) {

        let data = [];
        let target = [];


        //create nested arrays
        for (let i = 0; i < irisJS.getIrisNumClasses(); ++i) { //irisJS.getIrisNumClasses() should return 3
            data.push([]);
            target.push([]);
        }


        let irisData = [];
        /*
         * check convertedIrisData array. if empty use default iris data (nnworker.data.iris.js)
         */
        if(convertedIrisData.length === 0){ //use default values
            irisData = irisJS.getIrisData();
        }else{
            irisData = convertedIrisData;
        }

        /**
         * split iris data by species
         */
        irisData.forEach(function (element) {
            const species = element[element.length-1];
            data[species].push(element.slice(0));
            target[species].push(species);
        });

        let xTrains = [];
        let yTrains = [];
        let xTests  = [];
        let yTests  = [];

        for(let i = 0 ; i < irisJS.getIrisNumClasses(); i++){
            const [xTrain, yTrain, xTest, yTest] = toTernsors(data[i], target[i], testDataInPercent);
            xTrains.push(xTrain);
            yTrains.push(yTrain);
            xTests.push(xTest);
            yTests.push(yTest);
        }


        const axis = 0;

        //Concatenates tensors along one dimension.
        const concatenatedxTrains = tf.concat(xTrains, axis);
        const concatenatedyTrains = tf.concat(yTrains, axis);
        const concatenatedxTests  = tf.concat(xTests, axis);
        const concatenatedyTests  = tf.concat(yTests, axis);

        return [concatenatedxTrains, concatenatedyTrains, concatenatedxTests, concatenatedyTests];
    },
    
    csvToArray: function (csv) {

        let lines=csv.split("\n");

        let result = [];

        for(let i=0;i<lines.length;i++){
            let currentline = lines[i].split(",");
            let temp = [];

            if(currentline.length > 0 && currentline.length == 5) {
                temp.push(parseFloat(currentline[0]));
                temp.push(parseFloat(currentline[1]));
                temp.push(parseFloat(currentline[2]));
                temp.push(parseFloat(currentline[3]));
                temp.push(parseFloat(currentline[4]));

                result.push(temp);
            }
        }
        return result;
    }
};
