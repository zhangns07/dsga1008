require 'nn'
require 'image'
require 'cunn'
-- require 'unsup'
dofile '../provider.lua'

m = require 'manifold';

-- randomly chose 1000 labeled data from test dataset
provider = torch.load('../provider.t7')
testData = torch.load('testData.t7')
testset = torch.Tensor(1000,16,88,88)
testsetorig = torch.Tensor(1000,3,96,96)
labels = torch.Tensor(1000,1)
for i = 1,1000 do
	j =  math.ceil(torch.uniform(1e-12,8000))
	testset[{{i},{},{},{}}] = testData[j]
	testsetorig[{{i},{},{},{}}] = provider.testData.data[j]
	labels[i] = provider.testData.labels[j]
end 


-- load trained model
model = torch.load('logs.autoencoder60000.sample/model.net')


-- compute last layer feature
x = torch.DoubleTensor(1000,1024)
for i =1, 1000, 10 do
	outputs = model:forward(testset:narrow(1,i,10):cuda()):double()
	x[{{i,i+9},{}}] = outputs
end

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(x, opts)


mapped_x1:size()
im_size = 4096
-- map_im = m.draw_image_map(mapped_x1, x:resize(x:size(1), 1, 28, 28), im_size, 0, true)
map_im = m.draw_image_map(mapped_x1, x:resize(x:size(1), 1, 32,32), im_size, 0, true)
map_im = m.draw_image_map(mapped_x1, testsetorig, im_size, 0, true)

image.save('./autoencoder2.tSNE.jpg', map_im)


