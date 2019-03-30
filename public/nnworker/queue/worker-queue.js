let request = require('request');
let convert = require('xml-js');
let tf_iris_network_trainer = require('../tensorflow/iris/network-trainer');
let tf_mnist_network_trainer = require('../tensorflow/mnist/network-trainer');
let winston = require('winston');
let addresses = require('../util/addresses');




var options = {
    compact: true,
    spaces: 2,
    ignoreDeclaration: true,
    ignoreInstruction: true,
    ignoreAttributes: false,
    ignoreComment: true,
    ignoreCdata: true,
    ignoreDoctype: true

};


let queue = [];
let epochsFromUi = null;
let learningRateFromUi = null;

//const VINNSL_SERVICE_ENDPOINT = "http://localhost:8080";
//const VINNSL_SERVICE_ENDPOINT = "http://vinnsl-service:8080";



module.exports = {
    pushToQueue: function (id, epochs, learningRate, res){

        winston.log('info', 'id: ' + id + ' queued', {})

        epochsFromUi = epochs;
        learningRateFromUi = learningRate;
        queue.push(id);



         setInterval(function() {


            if(queue.length > 0){
                console.log("queue not empty: " + queue.length);


                let nnId = queue[0];

                request(addresses.getVinnslServiceEndpoint() + '/vinnsl/' + nnId, function (error, response, body) {
                    let json = convert.xml2json(body, options);
                    let vinnslJSON = JSON.parse(json);

                    let nnSpecies = getNNSpecies(vinnslJSON);

                    switch (nnSpecies) {
                        case "iris":
                            tf_iris_network_trainer.irisTrainer(vinnslJSON, epochsFromUi, learningRateFromUi, res);
                            break;
                        case "mnist":
                            tf_mnist_network_trainer.mnistTrainer(vinnslJSON);
                            break;
                        case "wine":
                            break;
                        default:
                            return;

                    }

                });

                queue.shift();




            }else{
               // console.log("queue is empty");
            }


        }, 5000);//method will be called every 5 second
    },
    getQueue: function () {
        return queue;
    }
};


function getNNSpecies(vinnslJSON) {
    if(vinnslJSON){
        if(vinnslJSON.vinnsl){
            if(vinnslJSON.vinnsl.description) {
                if (vinnslJSON.vinnsl.description.metadata) {
                    if (vinnslJSON.vinnsl.description.metadata.description) {
                        let description = vinnslJSON.vinnsl.description.metadata.description._text;

                        if(description.toLowerCase().includes("iris")){
                            return "iris";
                        }else if(description.toLowerCase().includes("mnist")){
                            return "mnist";
                        }else if(description.toLowerCase().includes("wine")){
                            return "wine";
                        }else{
                            return "";
                        }
                    }
                }
            }
        }
    }
}




