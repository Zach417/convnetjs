var unigrams = require('./unigrams');

module.exports = function (n, callback) {
  unigrams(function (unigrams) {
    var idx = 0;
    var ngrams = [];

    function add(gram) {
      idx++;
      for (var i = 0; i < unigrams.length; i++) {
        ngrams.push(gram + unigrams[i]);
      }

      idx++;
      if (idx < n) {
        for (var i = 0; i < unigrams.length; i++) {
          add(gram + unigrams[i]);
        }
      }
    }

    for (var i = 0; i < unigrams.length; i++) {
      idx = 0;
      ngrams.push(unigrams[i]);
      add(unigrams[i]);
    }

    callback(ngrams);
  });
}
