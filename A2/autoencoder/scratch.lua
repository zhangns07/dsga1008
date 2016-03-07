require 'nn'
require 'xlua'
require 'unsup'
require 'optim'
require 'cunn'

dofile '../provider.lua'
dofile './BatchFlip.lua'

provider = torch.load '../provider.t7'
unsupmodule = torch.load('model_60000.bin')
input =  provider.trainData.data[{{1,3},{},{},{}}]

-- forward working, params not working
model = nn.Sequential()
model:add(unsupmodule.encoder:float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/sample.lua'):cuda())
model:get(2).updateGradInput = function(input) return end

model:forward(input)
p,d = model:getParameters()

-- both working
model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/sample.lua'):cuda())
model:get(2).updateGradInput = function(input) return end

model:forward(input)
p,d = model:getParameters()

-- both working
model = nn.Sequential()
model:add(unsupmodule.encoder:float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())

model:forward(input)
p,d = model:getParameters()
