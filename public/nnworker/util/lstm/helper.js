const tf = require('@tensorflow/tfjs');




module.exports = {

    getChartSet: async function(text){
        let chartSet = [];
        for(let i = 0; i < text.length; ++i){
            if(chartSet.indexOf(text[i]) === -1){
                chartSet.push(text[i]);
            }
        }
        return chartSet;
    },
    converTextToIndices: async function(text, chartSet){
        let indices = new Uint16Array(textToIndices(text, chartSet));
        return indices;
    },
    generateBeginIndices: async function(textLength, sampleLength, sampleStep){
        let beginIndices = [];
        for(let i = 0; i <textLength-sampleLength-1; i += sampleStep){
            beginIndices.push(i);
        }

        tf.util.shuffle(beginIndices);
        return beginIndices;
    },
    getRandomSlice: async function(text, sampleLength, charSet){
        const startIdx = Math.round(Math.random() * (text.length - sampleLength -1));
        const textSlice = text.slice(startIdx, (startIdx+sampleLength));
        return [textSlice, textToIndices(text, charSet)];

    },
    sample: async function (prediction, temperature) {
        const logPreds = tf.div(tf.log(prediction), temperature);
        const expPreds = tf.exp(logPreds);
        const sumExpPreds = tf.sum(expPreds);
        prediction = tf.div(expPreds,  sumExpPreds);
        return tf.multinomial(prediction, 1, null, true).dataSync()[0];
    }

}

function textToIndices(text, charSet) {
    let indices = [];
    for(let i = 0; i < text.length; ++i){
        indices.push(charSet.indexOf(text[i]));
    }
    return indices;
}