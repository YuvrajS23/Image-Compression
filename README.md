# JPEG Image Compression Engine (CS663)

Yuvraj Singh 210050172 , Arnav Aditya Singh 210050018

## Running Instructions
To encode a file
```
python base.py encode -i <INPUT_FILE> -q <QUALITY_FACTOR> -o <OUTPUT_FILE(.pkl)>
```
To decode the output of above, use it as input in
```
python base.py decode -i <INPUT_FILE(.pkl)> -o <OUTPUT_FILE>
```
To test encode and decode,
```
python base.py encodeanddecode -i <INPUT_FILE> -q <QUALITY_FACTOR> -e <ENCODED_FILE(.pkl)> -o <OUTPUT_FILE>
```