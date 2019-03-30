const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const helper = require('./model-helper');
const {createCanvas, Image} = require('canvas');

const FILE_SAVE_PATH = 'public/nnworker/data/saved-models/mnist';


module.exports = {

    mnistTester : async function(imgData){

        try{

            if(fs.existsSync(FILE_SAVE_PATH+'/model.json')) {
                console.log('model exist... load model...');

                const canvas = createCanvas(280, 280);
                const ctx = canvas.getContext('2d');

                const img = new Image();
                img.src = imgData;
                ctx.drawImage(img, 0, 0);

                /**
                 * load model
                 */
                const model = await helper.loadMnistModel();

                let tfImage = tf.browser.fromPixels(canvas,1);
                let smallImageData = tf.image.resizeBilinear(tfImage, [28, 28]);
                smallImageData = tf.cast(smallImageData, 'float32');
                let tensor = smallImageData.expandDims(0);
                tensor = tensor.div(tf.scalar(255));
                const prediction = model.predict(tensor);
                const label = prediction.argMax(-1).dataSync();

                console.log('predicted value: '+ label);
                return label;

            }
        }catch (e) {
          console.log(e);
        }
    }


}