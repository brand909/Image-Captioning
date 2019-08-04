import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)          # freezes pre-trained model weights
        modules = list(resnet.children())[:-1]   # deletes the last fully-connected layer
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
       #with torch.no_grad():
       #    features = self.resnet(images)
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()  
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        #self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, features, captions):   
        
        captions_end_removed = captions[:,:-1]
        
        embeddings = self.embed(captions_end_removed)
        combined_inputs = torch.cat((features.unsqueeze(1), embeddings), 1) # Concatenates features and embeddings
        rnn_hidden_output, _ = self.lstm(combined_inputs, None)
        word_predictions = self.linear(rnn_hidden_output)         
        #rnn_hidden_output, _ = self.lstm(combined_inputs)
        #word_predictions = self.linear(rnn_hidden_output[0])
        
        return word_predictions
        

    def sample(self, inputs, rnn_state=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        remark = []
        stop_index = 1 # this is '<end>'
        
        for i in range(max_len):
                       
            rnn_hidden_output, rnn_state = self.lstm(inputs, rnn_state)
            word_predictions = self.linear(rnn_hidden_output)
            prediction = torch.argmax(word_predictions, dim=2)
            #prediction_values, predicted_index = torch.max(word_predictions, dim=2)
            predicted_index = prediction.item()
            remark.append(predicted_index)
            #predictions = torch.topk(word_predictions, k=beam_width, dim=2)      
            if predicted_index == stop_index:
                break      
            # Retrieve embeddings for next word based on inputs
            inputs = self.embed(prediction)
            
        return remark
        
        