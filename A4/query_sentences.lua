require 'nn'
require 'nngraph'
require('base')
stringx = require('pl.stringx')
require 'io'

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  return line
end

-- load model and dictionary
model = torch.load('model.net')
vocab_invmap = torch.load('vocab_invmap.t7')
vocab_map = torch.load('vocab_map.t7')
batch_size = 20

g_disable_dropout(model.rnns)
g_replace_table(model.s[0], model.start_s)

while true do
print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
	len = #line - 1
	len_to_gen = line[1]
	
	for i = 1,len do
		x = vocab_map[line[i+1]]
		-- if the word is not in dictionary, replace with <unk>
		if x == nil  then x = 27 end
		x = torch.Tensor(batch_size):fill(x)
		y = x

		-- update hidden state and get prediciton for y
	        perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
		pred = torch.exp(pred[1])
		pred = pred:resize(1,10000)

		-- sample a word y according to predicted distribution
		output_idx = torch.multinomial(pred,1)[1]
		output = vocab_invmap[output_idx[1]] 
	        g_replace_table(model.s[0], model.s[1])
	end
	
	
	for i = 1,len_to_gen do
		io.write(output," ")
		-- print(output)
		x = vocab_map[output]
		x = torch.Tensor(batch_size):fill(x)
		y = x
	        perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
		pred = torch.exp(pred[1])
		pred = pred:resize(1,10000)
		output_idx = torch.multinomial(pred,1)[1]
		output = vocab_invmap[output_idx[1]] 
	        g_replace_table(model.s[0], model.s[1])
	end
	io.write("\n")
  end
end


