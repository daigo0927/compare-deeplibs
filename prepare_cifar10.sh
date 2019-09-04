curl -OL https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
rm -f cifar-10-python.tar.gz
pip install numpy pillow
python extract_cifar10.py
rm -f cifar-10-batches-py/data_batch*
rm -f cifar-10-batches-py/test_batch*
mv cifar-10-batches-py cifar-10
