


const VINNSL_SERVICE_ENDPOINT = "http://localhost:8080";
const VINNSL_SERVICE_ENDPOINT_TF_JS = "http://localhost:8080/tensorflowJS";
const VINNSL_STORAGE_SERVICE_ENDPOINT = "http://localhost:8081/storage";
const VINNSL_UI_ENDPOINT ="http://localhost:8083/"

//const VINNSL_SERVICE_ENDPOINT = "http://vinnsl-service:8080/vinnsl";
//const VINNSL_SERVICE_ENDPOINT_TF_JS = "http://vinnsl-service:8080/tensorflowJS";
//const VINNSL_STORAGE_SERVICE_ENDPOINT = "http://vinnsl-storage-service:8081/storage";



module.exports = {

    getVinnslServiceEndpoint: function () {
        return VINNSL_SERVICE_ENDPOINT;
    },
    getVinnslTensorFlowJSEndpoint: function () {
        return VINNSL_SERVICE_ENDPOINT_TF_JS;
    },
    getVinnslStorageServiceEndpoint:function () {
        return VINNSL_STORAGE_SERVICE_ENDPOINT;
    },
    getVinnslUiEndpoint:function () {
        return VINNSL_UI_ENDPOINT;
    }

};