require 'xlua'
require 'unsup'
require 'optim'
require 'cunn'
dofile '../provider.lua'

-- transform data
unsupmodule = torch.load('model_60000.bin')
provider = torch.load '../provider.t7'

trainData = torch.Tensor(4000,16,88,88)
valData = torch.Tensor(1000,16,88,88)
testData = torch.Tensor(8000,16,88,88)

for i = 1,4000 do
	trainData[i] = unsupmodule.encoder(provider.trainData.data[i]:double() )
end

for i = 1,1000 do
	valData[i] = unsupmodule.encoder(provider.valData.data[i]:double() )
end

for i = 1,8000 do
	testData[i] = unsupmodule.encoder(provider.testData.data[i]:double() )
end

torch.save('trainData.t7',trainData)
torch.save('testData.t7',testData)
torch.save('valData.t7',valData)


