const fs = require('fs');
const util = require('util');
const zlib = require('zlib');
const https = require('https');
const tf = require('@tensorflow/tfjs');
const readFile = util.promisify(fs.readFile);
let request = require('request');



const BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES_ZIP_FILE = "train-images-idx3-ubyte.gz";
const TRAIN_LABELS_ZIP_FILE = "train-labels-idx1-ubyte.gz";
const TEST_IMAGES_ZIP_FILE  = "t10k-images-idx3-ubyte.gz";
const TEST_LABELS_ZIP_FILE  = "t10k-labels-idx1-ubyte.gz";

const TRAIN_IMAGES_FILE = "train-images-idx3-ubyte";
const TRAIN_LABELS_FILE = "train-labels-idx1-ubyte";
const TEST_IMAGES_FILE  = "t10k-images-idx3-ubyte";
const TEST_LABELS_FILE  = "t10k-labels-idx1-ubyte";




module.exports = {


    downloadTrainImageFile: async function() {
        return new Promise(resolve => {
            const url = BASE_URL + TRAIN_IMAGES_ZIP_FILE;
            if(fs.existsSync(TRAIN_IMAGES_FILE)){
                fs.readFileSync(TRAIN_IMAGES_FILE, function read(err, data) {
                    if(err){
                        throw err;
                    }
                    let test = data;
                })
                //resolve(readFile(TRAIN_IMAGES_FILE));
                return;
            }
            let file = fs.createWriteStream(TRAIN_IMAGES_FILE);
            console.log("Downloading "+TRAIN_IMAGES_FILE);
            request(url, async function (error, response, body) {
                const unziper = zlib.createGunzip();
                response.pipe(unziper).pipe(file);
                unziper.on('end', ()=>{
                    resolve(readFile(TRAIN_IMAGES_FILE));
                })
            });
        })

    },function(err){
        console.log(err);
    },

    downloadTrainLabelsFile: async function() {
        return new Promise(resolve => {
            const url = BASE_URL + TRAIN_LABELS_ZIP_FILE;
            if(fs.existsSync(TRAIN_LABELS_FILE)){
                resolve(readFile(TRAIN_LABELS_FILE));
                return;
            }
            let file = fs.createWriteStream(TRAIN_LABELS_FILE);
            console.log("Downloading "+TRAIN_LABELS_FILE);
            request(url, async function (error, response, body) {
                const unziper = zlib.createGunzip();
                response.pipe(unziper).pipe(file);
                unziper.on('end', ()=>{
                    resolve(readFile(TRAIN_LABELS_FILE));
                })
            });
        })
    },function(err){
        console.log(err);
    },

    downloadTestImagesFile: async function() {
        return new Promise(resolve => {
            const url = BASE_URL + TEST_IMAGES_ZIP_FILE;
            if(fs.existsSync(TEST_IMAGES_FILE)){
                resolve(readFile(TEST_IMAGES_FILE));
                return;
            }
            let file = fs.createWriteStream(TEST_IMAGES_FILE);
            console.log("Downloading "+TEST_IMAGES_FILE);
            request(url, async function (error, response, body) {
                const unziper = zlib.createGunzip();
                response.pipe(unziper).pipe(file);
                unziper.on('end', ()=>{
                    resolve(readFile(TEST_IMAGES_FILE));
                })
            });
        })
    },function(err){
        console.log(err);
    },

    downloadTestLabelsFile: async function() {
        return new Promise(resolve => {
            const url = BASE_URL + TEST_LABELS_ZIP_FILE;
            if(fs.existsSync(TEST_LABELS_FILE)){
                resolve(readFile(TEST_LABELS_FILE));
                return;
            }
            let file = fs.createWriteStream(TEST_LABELS_FILE);
            console.log("Downloading "+TEST_LABELS_FILE);
            request(url, async function (error, response, body) {
                const unziper = zlib.createGunzip();
                response.pipe(unziper).pipe(file);
                unziper.on('end', ()=>{
                    resolve(readFile(TEST_LABELS_FILE));
                })
            });
        })
    },function(err){
        console.log(err);
    }

};

