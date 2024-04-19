ROOT=".tmp"

mkdir $ROOT
cd $ROOT
git clone https://github.com/THUDM/ImageReward.git
echo "from .utils import *" > ImageReward/ImageReward/init.py
cp -r "ImageReward/ImageReward" ../

rm -rf $ROOT