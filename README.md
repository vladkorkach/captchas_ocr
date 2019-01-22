###captcha solver

Some samples for simples captchas reader
Current version of neural network contains three parts
CNN - convolution neural network for futures extraction
GRU (RNN based cells) - for sequences analysis
CTC - Final part for converting sequences to correct labels 

Currently application works as follows - opencv only reads file and gives it to neural network model

Neural network gives the correct variant of captcha.
The accuracy during training was 95%
