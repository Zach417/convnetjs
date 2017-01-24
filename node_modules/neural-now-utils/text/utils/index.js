module.exports = {
  countChar: function (searchStr, str) {
    var searchStrLen = searchStr.length;
    var startIndex = 0, index, indices = [];
    str = str.toLowerCase();
    searchStr = searchStr.toLowerCase();

    var count = 0;
    while ((index = str.indexOf(searchStr, startIndex)) > -1) {
        count++;
        startIndex = index + searchStrLen;
    }
    return count;
  },
}
