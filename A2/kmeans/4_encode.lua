require 'xlua'
require 'unsup'
require 'optim'
require 'cunn'
dofile '../provider.lua'

kernels = torch.load('kmeans_16.t7')


-- transform data
provider = torch.load '../provider.t7'
trainData = torch.Tensor(4000,16,88,88)
testData = torch.Tensor(8000,16,88,88)

model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3,16,9,9,1,1))
model.modules[1].weight = kernels:float()
model.modules[1].bias= torch.Tensor(100):zero()

for i = 1,4000 do
	trainData[i] = model:forward(provider.trainData.data[i])
end

for i = 1,8000 do
	testData[i] = model:forward(provider.testData.data[i])
end

torch.save('trainData.t7',trainData)
torch.save('testData.t7',testData)


