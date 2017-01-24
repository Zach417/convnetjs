var Utils = require('../index');
Utils.Text.toBigramVector("test", function (vector) {
  for (var i = 0; i < vector.length; i ++) {
    if (vector[i] > 0) {
      console.log(i, vector[i]);
    }
  }
  console.log(vector.length);
});
