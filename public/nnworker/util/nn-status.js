const axios = require('axios');
let addresses = require('./addresses');


const nnStatus = {
    QUEUED: 'queued',
    CREATED: 'created',
    INPROGRESS: 'inprogress',
    FINISHED: 'finished',
    ERROR: 'error'
}


module.exports = {

    setStatusToFinished: function(id) {
    axios.put(addresses.getVinnslServiceEndpoint() + '/status/' + id + '/FINISHED')
        .then(response => {
            //ignore response
        })
        .catch(error => {
            console.log(error);
        });
    },

    setStatusToInProgress: function(id) {
    axios.put(addresses.getVinnslServiceEndpoint() + '/status/' + id + '/INPROGRESS')
        .then(response => {
            //ignore response
        })
        .catch(error => {
            console.log(error);
        });
}
}
