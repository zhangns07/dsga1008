require 'nn'
require 'image'
require 'unsup'
dofile '../provider.lua'


-- kernel
kernels = torch.load('kmeans_16.t7')
ks = kernels:reshape(16,3,9,9)
de = image.toDisplayTensor{input=ks,
	padding=2,
	nrow=4,
	symmetric=true}
image.save('./kernel.jpg', de)


-- data
provider = torch.load('../provider.t7')
trainData_pre =  provider.trainData.data
trainData_pos = torch.load('trainData.t7')

image.save('./pre.jpg', trainData_pre[1])

pos = image.toDisplayTensor{input=trainData_pos[1],
	padding=2,
	nrow=4,
	symmetric=true}

image.save('./pos.jpg', pos)


