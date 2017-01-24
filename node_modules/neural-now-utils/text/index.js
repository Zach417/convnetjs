var ngrams = require('./ngrams');
var utils = require('./utils');

var textUtils = {
  ngrams: ngrams,
  toUnigramVector: function (corpus) {
    var unigrams = ngrams.unigrams;
    var vector = [];
    for (var i = 0; i < bigrams.length; i++) {
      var count = utils.countChar(bigrams[i], corpus);
      vector.push(count);
    }
    return vector;
  },
  toBigramVector: function (corpus) {
    var bigrams = ngrams.bigrams;
    var vector = [];
    for (var i = 0; i < bigrams.length; i++) {
      var count = utils.countChar(bigrams[i], corpus);
      vector.push(count);
    }
    return vector;
  },
}

module.exports = textUtils;
