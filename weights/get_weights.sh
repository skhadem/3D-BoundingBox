#! /bin/bash
echo "downloading 3d bbox weights ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA" -O epoch_10.pkl && rm -rf /tmp/cookies.txt
echo "downloading yolo weights ..."
wget https://pjreddie.com/media/files/yolov3.weights
