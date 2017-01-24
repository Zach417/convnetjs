var fs = require('fs');
var ngrams = require('../ngrams');
var utils = require('../utils');
var corpus = fs.readFileSync(__dirname + "/corpus.txt", "utf-8");
ngrams.bigrams(function (bigrams) {
  var vector = [];
  for (var i = 0; i < bigrams.length; i++) {
    var count = utils.countChar(bigrams[i], corpus);
    vector.push(count);
  }
});
