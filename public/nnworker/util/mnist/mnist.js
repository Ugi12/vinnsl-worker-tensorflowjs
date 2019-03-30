let mnistFetch = require('./mnist-fetcher');
const tf = require('@tensorflow/tfjs-node');

/******/
const fs = require('fs');
const util = require('util');
const zlib = require('zlib');
const https = require('https');
const readFile = util.promisify(fs.readFile);
const assert = require('assert');

const BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES_ZIP_FILE = "train-images-idx3-ubyte.gz";
const TRAIN_LABELS_ZIP_FILE = "train-labels-idx1-ubyte.gz";
const TEST_IMAGES_ZIP_FILE  = "t10k-images-idx3-ubyte.gz";
const TEST_LABELS_ZIP_FILE  = "t10k-labels-idx1-ubyte.gz";

const TRAIN_IMAGES_FILE = "train-images-idx3-ubyte";
const TRAIN_LABELS_FILE = "train-labels-idx1-ubyte";
const TEST_IMAGES_FILE  = "t10k-images-idx3-ubyte";
const TEST_LABELS_FILE  = "t10k-labels-idx1-ubyte";

const MNIST_SAVE_PATH = `${__dirname}/../../data/mnist/`;


/*********/

//MNIST values
const image_height = 28;
const image_width  = 28;
const image_size = 28 * 28;
const number_species = 10 // 0-9
const total_dataset_elements = 65000;
const number_train_elements = 55000;
const number_test_elements = total_dataset_elements - number_train_elements;
const image_header_bytes = 16;
const label_header_bytes = 8;
const label_record_byte = 1;
const image_header_magic_num = 2051;


let dataset = null;
let trainSize = 0;
let testSize = 0;
let trainBatchIndex = 0;
let testBatchIndex = 0;


module.exports = {

    getData: async function () {


        //load data if null
        if(dataset == null){
            dataset = await Promise.all([loadTrainImages(), loadTrainLabels(), loadTestImages(), loadTestLabels()]);
            trainSize = dataset[0].length;
            testSize  = dataset[2].length;
        }

        const trainDataImagesIndex = 0;
        const trainDataLabelsIndex = 1;
        const testDataImagesIndex  = 2;
        const testDataLabelsIndex  = 3;

        tf.util.assert(dataset[trainDataImagesIndex].length === dataset[trainDataLabelsIndex].length, 'mismatch training data (images and labels)');
        tf.util.assert(dataset[testDataImagesIndex].length === dataset[testDataLabelsIndex].length, 'mismatch test data (images and labels)');

        const trainImagesShape = [dataset[trainDataImagesIndex].length, image_height, image_width, 1];
        const testImagesShape  = [dataset[testDataImagesIndex].length, image_height, image_width, 1];


        const trainImages = new Float32Array(tf.util.sizeFromShape(trainImagesShape));
        const testImages = new Float32Array(tf.util.sizeFromShape(testImagesShape));

        const trainLabels = new Int32Array(tf.util.sizeFromShape([dataset[trainDataImagesIndex].length, 1]));
        const testLabels = new Int32Array(tf.util.sizeFromShape([dataset[testDataImagesIndex].length, 1]));

        const tttt = tf.util.sizeFromShape(trainImagesShape);
        let trainImageOffset = 0;
        let trainLabelsOffset = 0;
        //for loop traindata
        for(let i = 0; i < dataset[trainDataImagesIndex].length; ++i){
            trainImages.set(dataset[trainDataImagesIndex][i], trainImageOffset);
            trainLabels.set(dataset[trainDataLabelsIndex][i], trainLabelsOffset);
            trainImageOffset += image_height * image_width;
            trainLabelsOffset += 1;
        }

        let testImageOffset = 0;
        let testLabelsOffset = 0;
        //for loop testdata
        for(let i = 0; i < dataset[testDataImagesIndex].length; ++i){
            testImages.set(dataset[testDataImagesIndex][i], testImageOffset);
            testLabels.set(dataset[testDataLabelsIndex][i], testLabelsOffset);
            testImageOffset += image_height * image_width;
            testLabelsOffset += 1;
        }


        const trainImgs = tf.tensor4d(trainImages, trainImagesShape);
        const trainLbls = tf.oneHot(tf.tensor1d(trainLabels, 'int32'), number_species).toFloat();
        const testImgs  = tf.tensor4d(testImages, testImagesShape);
        const testLbls  = tf.oneHot(tf.tensor1d(testLabels, 'int32'), number_species).toFloat();

        return [trainImgs, trainLbls, testImgs, testLbls];


    }

}

async function loadTrainImages() {
    const buffer = await downloadTrainImageFile();
    return await loadImages(buffer);

}
async function loadTestImages() {
    const buffer = await downloadTestImagesFile();
    return await loadImages(buffer);
}

async function loadTrainLabels() {
    const buffer = await downloadTrainLabelsFile();
    return await loadLabels(buffer);
}

async function loadTestLabels() {
    const buffer = await downloadTestLabelsFile();
    return await loadLabels(buffer);

}

async function loadImages(buffer) {

    const headerBytes = image_header_bytes;
    const recordBytes = image_height * image_width;

    const headerValues = loadHeaderValues(buffer, headerBytes);
    assert.equal(headerValues[0], 2051);
    assert.equal(headerValues[2], image_height);
    assert.equal(headerValues[3], image_width);

    const images = [];
    let index = headerBytes;
    while(index < buffer.byteLength){
        const array = new Float32Array(recordBytes);
        for(let i = 0; i < recordBytes; ++i){
            array[i] = buffer.readUInt8(index++) / 255;
        }
        images.push(array);
    }
    assert.equal(images.length, headerValues[1]);
    return images;
}


async function loadLabels(buffer) {
    const headerBytes = label_header_bytes;
    const recordBytes = label_record_byte;
    const headerValues = loadHeaderValues(buffer, headerBytes);
    assert.equal(headerValues[0], 2049, "HeaderValues[0] should have 2049 is: " + headerValues[0]);

    const labels = [];
    let index = headerBytes;

    while(index < buffer.byteLength){
        const array = new Int32Array(recordBytes);
        for(let i = 0; i < recordBytes; ++i){
            array[i] = buffer.readUInt8(index++);
        }
        labels.push(array);
    }

    assert.equal(labels.length, headerValues[1], "length of the labels and the length of headerValues[1] should be the same. " +
        "(label.length:"+labels.length+" and the headerValues[1]:"+headerValues[1]);

    return labels;

}

function loadHeaderValues(buffer, headerLength) {
    const headerValues = [];
    for(let i = 0; i < headerLength/4; ++i){
        headerValues[i] = buffer.readUInt32BE(i * 4);
    }
    return headerValues;
}






async function downloadTrainImageFile() {
    return new Promise(resolve => {
        const url = BASE_URL + TRAIN_IMAGES_ZIP_FILE;
        if(fs.existsSync(MNIST_SAVE_PATH+TRAIN_IMAGES_FILE)){
            resolve(readFile(MNIST_SAVE_PATH+TRAIN_IMAGES_FILE));
            return;
        }else {

            let file = fs.createWriteStream(MNIST_SAVE_PATH + TRAIN_IMAGES_FILE);
            console.log("Downloading " + TRAIN_IMAGES_FILE);
            https.get(url, (res) => {
                const unziper = zlib.createGunzip();
                res.pipe(unziper).pipe(file);
                unziper.on('end', () => {
                    resolve(readFile(MNIST_SAVE_PATH + TRAIN_IMAGES_FILE));
                    return;
                });
            });
        }
    });

}

async function downloadTrainLabelsFile() {
    return new Promise(resolve => {
        const url = BASE_URL + TRAIN_LABELS_ZIP_FILE;
        if(fs.existsSync(MNIST_SAVE_PATH + TRAIN_LABELS_FILE)){
            resolve(readFile(MNIST_SAVE_PATH + TRAIN_LABELS_FILE));
            return;
        }else{
            let file = fs.createWriteStream(MNIST_SAVE_PATH + TRAIN_LABELS_FILE);
            console.log("Downloading " + TRAIN_LABELS_FILE);
            https.get(url, (res) => {
                const unziper = zlib.createGunzip();
                res.pipe(unziper).pipe(file);
                unziper.on('end', ()=>{
                    resolve(readFile(MNIST_SAVE_PATH + TRAIN_LABELS_FILE));
                    return;
                })
            });
        }
    });
}

async function downloadTestImagesFile() {
    return new Promise(resolve => {
        const url = BASE_URL + TEST_IMAGES_ZIP_FILE;
        if(fs.existsSync(MNIST_SAVE_PATH+TEST_IMAGES_FILE)){
            resolve(readFile(MNIST_SAVE_PATH+TEST_IMAGES_FILE));
            return;
        }else{
            let file = fs.createWriteStream(MNIST_SAVE_PATH+TEST_IMAGES_FILE);
            console.log("Downloading "+TEST_IMAGES_FILE);
            https.get(url, (res) => {
                const unziper = zlib.createGunzip();
                res.pipe(unziper).pipe(file);
                unziper.on('end', ()=>{
                    resolve(readFile(MNIST_SAVE_PATH+TEST_IMAGES_FILE));
                    return;
                })
            });
        }
    });
}
async function downloadTestLabelsFile() {
    return new Promise(resolve => {
        const url = BASE_URL + TEST_LABELS_ZIP_FILE;
        if(fs.existsSync(MNIST_SAVE_PATH+TEST_LABELS_FILE)){
            resolve(readFile(MNIST_SAVE_PATH+TEST_LABELS_FILE));
            return;
        }else{
            let file = fs.createWriteStream(MNIST_SAVE_PATH+TEST_LABELS_FILE);
            console.log("Downloading "+TEST_LABELS_FILE);
            https.get(url, (res) => {
                const unziper = zlib.createGunzip();
                res.pipe(unziper).pipe(file);
                unziper.on('end', ()=>{
                    resolve(readFile(MNIST_SAVE_PATH+TEST_LABELS_FILE));
                    return;
                })
            });
        }
    });
}