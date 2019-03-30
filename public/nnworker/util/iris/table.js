const irisJS = require('../../data/iris/iris');
const axios = require('axios');
let addresses = require('../addresses');

module.exports = {

    renderResultTable: async function renderResultTable(xData, yTrue, yPred, predictOut, res, id) {


        let result = [];
        let data = {};
        let correctPredicted = 0;
        yTrue.forEach(function (value, index) {

            data = {};
            let petal_len = '';
            let petal_wid = '';
            let sepal_len = '';
            let sepal_wid = '';

            for(let i = 0; i < 4; ++i){
                switch (i) {
                    case 0:
                        petal_len = xData[5 * index + i].toFixed(1);
                        break;
                    case 1:
                        petal_wid = xData[5 * index + i].toFixed(1);
                        break;
                    case 2:
                        sepal_len = xData[5 * index + i].toFixed(1);
                        break;
                    case 3:
                        sepal_wid = xData[5 * index + i].toFixed(1);
                        break;
                    default:
                        break;
                }
               a =9;
            }

            let IRIS_CLASSES = irisJS.getIrisClasses();
            let IRIS_NUM_CLASSES = irisJS.getIrisNumClasses();

            let originIrisClass = IRIS_CLASSES[yTrue[index]];
            let predictedIrisClass = IRIS_CLASSES[yPred[index]];
            let correct_pred = originIrisClass === predictedIrisClass ? 1 : 0;
            if(correct_pred == 1){
                correctPredicted++
            }


            const probabilities =
                predictOut.slice(index * IRIS_NUM_CLASSES, (index + 1) * IRIS_NUM_CLASSES);

            let setosaInPercent = probabilities[0] * 100;
            let versicolorInPercent = probabilities[1] * 100;
            let virginicaInPercent = probabilities[2] * 100;

            //cell data
            data = {
                petal_length: petal_len,
                petal_width:  petal_wid,
                sepal_length: sepal_len,
                sepal_width:  sepal_wid,
                origin_iris: originIrisClass,
                predicted_iris: predictedIrisClass,
                correct_prediction: correct_pred,
                setosa_in_percent: setosaInPercent.toFixed(2),
                versicolor_in_percent: versicolorInPercent.toFixed(2),
                virginica_in_percent: virginicaInPercent.toFixed(2),

            };

            result.push(data);

        })

        axios.put(addresses.getVinnslServiceEndpoint() + '/status/' + id + '/FINISHED')
            .then(response => {
                //ignore response
            })
            .catch(error => {
                console.log(error);
            });

        let responseObj = [];
        let predInPercent = correctPredicted / yTrue.length * 100;
        predInPercent = predInPercent.toFixed(2);

        if(result.length > 0){
            responseObj.push(result);
            responseObj.push(predInPercent);
            res.send(responseObj);
        }
        return predInPercent;

    },function(err){
        console.log(err);
    }


};