require 'nn'
require 'image'
require 'unsup'
dofile '../provider.lua'


-- autoencoders
-- filter
module = torch.load('model_60000.bin')

eweight = module.encoder.modules[1].weight
de = image.toDisplayTensor{input=eweight,
	padding=2,
	nrow=3,
	symmetric=true}
image.save('./filters_enc.jpg', de)


-- augmentation
provider = torch.load('../provider.t7')
trainData_pre =  provider.trainData.data
trainData_pos = torch.load('trainData.t7')

image.save('./pre.jpg', trainData_pre[1])

pos = image.toDisplayTensor{input=trainData_pos[1],
	padding=2,
	nrow=4,
	symmetric=true}

image.save('./pos.jpg', pos)


