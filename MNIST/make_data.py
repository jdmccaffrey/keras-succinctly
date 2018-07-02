# make_data.py
# raw binary MNIST to Keras text file
#
# go to http://yann.lecun.com/exdb/mnist/ and
# download the four g-zipped files:
# train-images-idx3-ubyte.gz (60,000 train images) 
# train-labels-idx1-ubyte.gz (60,000 train labels) 
# t10k-images-idx3-ubyte.gz  (10,000 test images) 
# t10k-labels-idx1-ubyte.gz  (10,000 test labels) 
# 
# use the 7-Zip program to unzip the four files.
# I recommend adding a .bin extension to remind
# you they're in a proprietary binary format
#
# run the script twice, once for train data, once for
# test data, changing the file names as appropriate.
# uses pure Python only

# target format:
# 5 ** 0 0 152 27 .. 0
# 7 ** 0 0 38 122 .. 0
# label digit at [0]    784 vals at [2-786]
# dummy ** seperator at [1] 

def generate(img_bin_file, lbl_bin_file,
            result_file, n_images):

  img_bf = open(img_bin_file, "rb")    # binary image pixels
  lbl_bf = open(lbl_bin_file, "rb")    # binary labels
  res_tf = open(result_file, "w")      # result file

  img_bf.read(16)   # discard image header info
  lbl_bf.read(8)    # discard label header info

  for i in range(n_images):   # number images requested 
    # digit label first
    lbl = ord(lbl_bf.read(1))  # get label like '3' (one byte)
    res_tf.write(str(lbl))

    # encoded = [0] * 10         # make one-hot vector
    # encoded[lbl] = 1
    # for i in range(10):
    #  res_tf.write(str(encoded[i]))
    #  res_tf.write(" ")  # like 0 0 0 1 0 0 0 0 0 0 

    res_tf.write(" ** ")  # arbitrary seperator char for readibility

    # now do the image pixels
    for j in range(784):  # get 784 vals for each image file
      val = ord(img_bf.read(1))
      res_tf.write(str(val))
      if j != 783: res_tf.write(" ")  # avoid trailing space 
    res_tf.write("\n")  # next image

  img_bf.close(); lbl_bf.close();  # close the binary files
  res_tf.close()                   # close the result text file

# ================================================================

def main():
  # generate(".\\UnzippedBinary\\train-images.idx3-ubyte.bin",
  #         ".\\UnzippedBinary\\train-labels.idx1-ubyte.bin",
  #         ".\\mnist_train_keras_1000.txt",
  #         n_images = 1000)  # first n images

  generate(".\\UnzippedBinary\\t10k-images.idx3-ubyte.bin",
          ".\\UnzippedBinary\\t10k-labels.idx1-ubyte.bin",
          ".\\mnist_test_keras_foo.txt",
          n_images = 100)  # first n images

if __name__ == "__main__":
  main()
