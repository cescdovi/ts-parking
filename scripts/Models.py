import torch
from torch import nn

NUM_EPOCHS = 2
BATCH_SIZE = 64
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT_RATE = 0.0

INPUT_SIZE = 1
OUTPUT_SIZE = 1
WINDOW_INPUT_LENGTH = 24
WINDOW_OUTPUT_LENGTH = 1

class VanillaRNN(nn.Module):
    def __init__(self,
                 input_size = INPUT_SIZE,
                 hidden_size = HIDDEN_SIZE,
                 output_size = OUTPUT_SIZE,
                 num_layers = NUM_LAYERS,
                 dropout = None):
        super().__init__()


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = 0.0 if dropout is None else dropout

        #aplicar cuello de botella al nº de neuronas por capa en siguientes capas
        self.neurons_per_layer = [self.hidden_size // (2**i) for i in range(self.num_layers)]

        #CREAR BLOQUES: RNN + DROPOUT
        self.blocks = nn.ModuleList() #crear lista para almacenar bloques de pytorch (lstm + dropout)

        for i in range(num_layers):
            in_size = input_size if i == 0 else self.neurons_per_layer[i-1] #calcular de forma dinamica el tamaño de entrada a cada capa
            out_size = self.neurons_per_layer[i]
            self.blocks.append(nn.RNN(input_size= in_size,hidden_size=out_size, batch_first= True)) #añadir capa RNN

            #añadir dropout en todas las capas menos la última
            if i < num_layers-1:
                self.blocks.append(nn.Dropout(p = self.dropout))

        #capa final de salida
        self.fc = nn.Linear(in_features= self.neurons_per_layer[-1],
                            out_features = self.output_size)
        
    def forward(self, x):
        batch_size = x.size(0) #tamaño del batch
        out = x #reasignar datos de entrada
        block_idx = 0

        for i in range(self.num_layers):
            rnn = self.blocks[block_idx]

            #inicializar celda de memoria y estado oculto aleatoriamente
            h0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(x.device)

            # out: all_hidde_states
            out, _ = rnn(out, h0)

            #actualizar indice para saltar dropout
            block_idx += 1

            # Aplicar dropout si corresponde
            if block_idx < len(self.blocks) and isinstance(self.blocks[block_idx], nn.Dropout):
                out = self.blocks[block_idx](out)
                block_idx += 1

        final_output = out[:, -1, :]  # Último paso temporal
        return self.fc(final_output)


######################################################################


class VanillaLSTM(nn.Module):
    def __init__(self,
                input_size = INPUT_SIZE, 
                hidden_size = HIDDEN_SIZE, 
                output_size = OUTPUT_SIZE, 
                num_layers = NUM_LAYERS, 
                dropout = DROPOUT_RATE):
        super().__init__()

        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        #aplicar cuello de botella al nº de neuronas por capa en siguientes capas
        self.neurons_per_layer = [self.hidden_size // (2**i) for i in range(self.num_layers)]

        #CREAR BLOQUES: LSTM + DROPOUT
        self.blocks = nn.ModuleList() #crear lista para almacenar bloques de pytorch (lstm + dropout)

        for i in range(num_layers):
            in_size = input_size if i == 0 else self.neurons_per_layer[i-1] #calcular de forma dinamica el tamaño de entrada a cada capa
            out_size = self.neurons_per_layer[i]
            self.blocks.append(nn.LSTM(input_size= in_size,hidden_size=out_size, batch_first= True)) #añadir capa LSTM

            #añadir dropout en todas las capas menos la última
            if i < num_layers-1:
                self.blocks.append(nn.Dropout(p = self.dropout))

        #capa final de salida
        self.fc = nn.Linear(in_features= self.neurons_per_layer[-1],
                            out_features = self.output_size)
        
        
    def forward(self, x):
        batch_size = x.size(0) #tamaño del batch
        out = x #reasignar datos de entrada
        block_idx = 0

        for i in range(self.num_layers):
            lstm = self.blocks[block_idx]

            #inicializar celda de memoria y estado oculto aleatoriamente
            h0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(x.device)
            c0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(x.device)

            # out: all_hidde_states
            out, _ = lstm(out, (h0, c0))

            #actualizar indice para saltar dropout
            block_idx += 1

            # Aplicar dropout si corresponde
            if block_idx < len(self.blocks) and isinstance(self.blocks[block_idx], nn.Dropout):
                out = self.blocks[block_idx](out)
                block_idx += 1

        final_output = out[:, -1, :]  # Último paso temporal
        return self.fc(final_output)

######################################################################

class VanillaGRU(nn.Module):
    def __init__(self,
                input_size = INPUT_SIZE, 
                hidden_size = HIDDEN_SIZE, 
                output_size = OUTPUT_SIZE, 
                num_layers = NUM_LAYERS, 
                dropout = None):
        super().__init__()

        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = 0.0 if dropout is None else dropout

        #aplicar cuello de botella al nº de neuronas por capa en siguientes capas
        self.neurons_per_layer = [self.hidden_size // (2**i) for i in range(self.num_layers)]

        #CREAR BLOQUES: GRU + DROPOUT
        self.blocks = nn.ModuleList() #crear lista para almacenar bloques de pytorch (lstm + dropout)

        for i in range(num_layers):
            in_size = input_size if i == 0 else self.neurons_per_layer[i-1] #calcular de forma dinamica el tamaño de entrada a cada capa
            out_size = self.neurons_per_layer[i]
            self.blocks.append(nn.GRU(input_size= in_size,hidden_size=out_size, batch_first= True)) #añadir capa GRU

            #añadir dropout en todas las capas menos la última
            if i < num_layers-1:
                self.blocks.append(nn.Dropout(p = self.dropout))

        #capa final de salida
        self.fc = nn.Linear(in_features= self.neurons_per_layer[-1],
                            out_features = self.output_size)
        
        
    def forward(self, x):
        batch_size = x.size(0) #tamaño del batch
        out = x #reasignar datos de entrada
        block_idx = 0

        for i in range(self.num_layers):
            gru = self.blocks[block_idx]

            #inicializar celda de memoria y estado oculto aleatoriamente
            h0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(x.device)
            
            # out: all_hidde_states
            out, _ = gru(out, h0)

            #actualizar indice para saltar dropout
            block_idx += 1

            # Aplicar dropout si corresponde
            if block_idx < len(self.blocks) and isinstance(self.blocks[block_idx], nn.Dropout):
                out = self.blocks[block_idx](out)
                block_idx += 1

        final_output = out[:, -1, :]  # Último paso temporal
        return self.fc(final_output)
