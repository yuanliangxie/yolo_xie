import numpy as np

##############用numpy实现了one_hot编码函数###################################################

def np_one_hot(array, num_classes):#假设分类是从1开始算起，而不是从0开始算起！
    squeeze_a = np.squeeze(array)
    array_len = len(squeeze_a)
    one_hot = np.zeros(shape = (array_len, num_classes))
    one_hot[np.arange(array_len), squeeze_a-1] = 1
    return one_hot

##################函数测试代码#################################################################

label = np.array([2,3,4,3,2,5,1])
num_class = 5
one_hot = np_one_hot(label, num_class)
print(one_hot)

#######################################################################################
