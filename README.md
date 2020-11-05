# AUPROTK

Requirements are provided in the requirements.txt file. So you can basically install them using pip freeze > requirements.txt in your system or in a new virtualenv.

Code has all the functionalities to extract stylometric features from text and even to classify texts using sklearn classifiers. As it is now, it extracts stylometric features of the input texts and provides the output in a file. 

A sample input is provided in input/

The only restriction is that the id (in the current setup tweet_id), needs to be unique.

A sample output is provided in output/

To run it, just do:

python3 main.py path/to/your/input path/to/output/file/that/the/program/creates

so, for instance,

python3 main ./input/sampleInput.txt ./output/sampleOutput.txt

The output is a correct jsonArray, so you can just load it with json.load and in each position of the array, you will have an object with the features in the "features" key, and the text id in the "tweet_id" key.