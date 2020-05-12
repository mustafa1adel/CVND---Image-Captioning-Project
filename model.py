import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first= True)
        self.fc = nn.Linear(hidden_size, vocab_size)        
        
    def forward(self, features, captions):
        batch_size = captions.shape[1]
        
        captions = captions[:, :-1]
        embeddings = self.embed(captions)

        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        output, hidden = self.lstm(inputs)
        return self.fc(output)
        
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        caption = []
        
        for i in range(max_len) :
            output, states = self.lstm(inputs, states)
            output = self.fc(output.squeeze(1))
            # get the index of highest class score
            out_max = output.argmax(dim=1)
            # extract the index from tensor
            predicted_word = out_max.item()
            caption.append(predicted_word)
            # break when get end tag
            if (predicted_word == 1):
                break
            # embedding the output as a next input
            inputs = self.embed(out_max).unsqueeze(1)
            
        return caption