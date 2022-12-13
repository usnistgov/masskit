# RESTful api for retention index calculation

## Format of the url
* [http://10.208.85.87/molecule/smiles/CCCc1cccc(c1)\[N%2B\](\[O-\])=O,c1ccccc1/ri/json](http://10.208.85.87/molecule/smiles/CCCc1cccc(c1)[N%2B]([O-])=O,c1ccccc1/ri/json)
  * the smiles string is encoded between the backslashes after "smiles"
  * some characters must encoded in the url
    * `\` -> `%5C`
    * `/` -> `%2F`
    * `+` -> `%2B`
    * `#` -> `%23`
    * this is called url encoding and functions to do this can be found in many libraries
      * python: `import urllib` then `urllib.parse.quote(url)`
      * c++ on windows: use [UrlEscapeA()](https://docs.microsoft.com/en-us/windows/win32/api/shlwapi/nf-shlwapi-urlescapea) or similar functions.
  * a list of smiles string can be sent, delimited by commas, as long as the url doesn't exceed 1000 characters
  * the smiles are standardized in the processing.  If they cannot be standardized, they are not include in the output.
## Format of the output
* the output is formatted in json
* note that backslashes are escaped in json, e.g. `\\`.  json parsing libraries handle this automatically
  * an easy to use, header only c++ json library is [JSON for Modern c++](https://github.com/nlohmann/json).  You only need to #include [json.hpp](https://github.com/nlohmann/json/releases/download/v3.8.0/json.hpp)
* errors are returned as http error codes, returning json formatted errors such as `{"error": "unable to compute ri for ['CCCC', 'c1ccccc1']"}`
## Access
* this is currently only available on the NIST network
* please limit request to one at a time (if need be, we can add more computers to handle many more requests)