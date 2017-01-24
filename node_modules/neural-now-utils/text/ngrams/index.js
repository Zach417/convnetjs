var unigrams = require('./unigrams');
var bigrams = require('./bigrams');
var ngrams = require('./ngrams');

module.exports = {
  unigrams: unigrams,
  bigrams: bigrams,
  generate: ngrams,
}
