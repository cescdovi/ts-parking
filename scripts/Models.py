import torch
from torch import nn

NUM_EPOCHS = 2
BATCH_SIZE = 64
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT_RATE = 0.0

INPUT_NUMERICAL_SIZE = 1
OUTPUT_SIZE = 1
WINDOW_INPUT_LENGTH = 24
WINDOW_OUTPUT_LENGTH = 1
MONTH_EMBEDDING_SIZE = 12
MONTH_EMBEDDING_PROJECTION_SIZE = 2
WEEKDAY_EMBEDDING_SIZE = 7
WEEKDAY_EMBEDDING_PROJECTION_SIZE = 2
class VanillaRNN(nn.Module):
    def __init__(self,
                 input_size = INPUT_NUMERICAL_SIZE,
                 hidden_size = HIDDEN_SIZE,
                 output_size = OUTPUT_SIZE,
                 num_layers = NUM_LAYERS,
                 month_embedding_size = MONTH_EMBEDDING_SIZE,
                 month_embedding_projection_size = MONTH_EMBEDDING_PROJECTION_SIZE,
                 weekday_embedding_size = WEEKDAY_EMBEDDING_SIZE,
                 weekday_embedding_projection_size = WEEKDAY_EMBEDDING_PROJECTION_SIZE,
                 dropout = None):
        super().__init__()


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = 0.0 if dropout is None else dropout

        self.embedding_mes = nn.Embedding(month_embedding_size, month_embedding_projection_size)
        self.embedding_weekday = nn.Embedding(weekday_embedding_size, weekday_embedding_projection_size)

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
        self.fc = nn.Linear(in_features= self.neurons_per_layer[-1]+month_embedding_projection_size + weekday_embedding_projection_size + 1,  # +1 para el is_weekend
                            out_features = self.output_size)
        
    def forward(self, seq24, month, weekday, holiday):
        batch_size = seq24.size(0) #tamaño del batch
        out = seq24 #reasignar datos de entrada
        block_idx = 0

        for i in range(self.num_layers):
            rnn = self.blocks[block_idx]
            #inicializar celda de memoria y estado oculto aleatoriamente
            h0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(seq24.device)
            # out: all_hidde_states
            out, _ = rnn(out, h0)
            #actualizar indice para saltar dropout
            block_idx += 1
            # Aplicar dropout si corresponde
            if block_idx < len(self.blocks) and isinstance(self.blocks[block_idx], nn.Dropout):
                out = self.blocks[block_idx](out)
                block_idx += 1
        

        last_hidden_size = out[:, -1, :]  # Último paso temporal
        output_month = self.embedding_mes(month).squeeze(1)
        output_weekday = self.embedding_weekday(weekday).squeeze(1)

        final_output = torch.cat((last_hidden_size, output_month, output_weekday, holiday), dim=1)

        return self.fc(final_output)



######################################################################


class VanillaLSTM(nn.Module):
    def __init__(self,
                input_size = INPUT_NUMERICAL_SIZE, 
                hidden_size = HIDDEN_SIZE, 
                output_size = OUTPUT_SIZE, 
                num_layers = NUM_LAYERS, 
                month_embedding_size = MONTH_EMBEDDING_SIZE,
                month_embedding_projection_size = MONTH_EMBEDDING_PROJECTION_SIZE,
                weekday_embedding_size = WEEKDAY_EMBEDDING_SIZE,
                weekday_embedding_projection_size = WEEKDAY_EMBEDDING_PROJECTION_SIZE,
                dropout = None):
        super().__init__()

        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = 0.0 if dropout is None else dropout


        self.embedding_mes = nn.Embedding(month_embedding_size, month_embedding_projection_size)
        self.embedding_weekday = nn.Embedding(weekday_embedding_size, weekday_embedding_projection_size)


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
        self.fc = nn.Linear(in_features= self.neurons_per_layer[-1]+month_embedding_projection_size + weekday_embedding_projection_size + 1,
                            out_features = self.output_size)
        
        
    def forward(self, seq24, month, weekday, holiday):
        batch_size = seq24.size(0) #tamaño del batch
        out = seq24 #reasignar datos de entrada
        block_idx = 0

        for i in range(self.num_layers):
            lstm = self.blocks[block_idx]

            #inicializar celda de memoria y estado oculto aleatoriamente
            h0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(seq24.device)
            c0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(seq24.device)

            # out: all_hidde_states
            out, _ = lstm(out, (h0, c0))

            #actualizar indice para saltar dropout
            block_idx += 1

            # Aplicar dropout si corresponde
            if block_idx < len(self.blocks) and isinstance(self.blocks[block_idx], nn.Dropout):
                out = self.blocks[block_idx](out)
                block_idx += 1

        last_hidden_size = out[:, -1, :]  # Último paso temporal
        output_month = self.embedding_mes(month).squeeze(1)
        output_weekday = self.embedding_weekday(weekday).squeeze(1)

        final_output = torch.cat((last_hidden_size, output_month, output_weekday, holiday), dim=1)

        return self.fc(final_output)
        

######################################################################

class VanillaGRU(nn.Module):
    def __init__(self,
                input_size = INPUT_NUMERICAL_SIZE, 
                hidden_size = HIDDEN_SIZE, 
                output_size = OUTPUT_SIZE, 
                num_layers = NUM_LAYERS,
                month_embedding_size = MONTH_EMBEDDING_SIZE,
                month_embedding_projection_size = MONTH_EMBEDDING_PROJECTION_SIZE,
                weekday_embedding_size = WEEKDAY_EMBEDDING_SIZE,
                weekday_embedding_projection_size = WEEKDAY_EMBEDDING_PROJECTION_SIZE,
                dropout = None):
        super().__init__()

        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = 0.0 if dropout is None else dropout

        self.embedding_mes = nn.Embedding(month_embedding_size, month_embedding_projection_size)
        self.embedding_weekday = nn.Embedding(weekday_embedding_size, weekday_embedding_projection_size)


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
        self.fc = nn.Linear(in_features= self.neurons_per_layer[-1]+month_embedding_projection_size + weekday_embedding_projection_size + 1,
                            out_features = self.output_size)
        
        
    def forward(self, seq24, month, weekday, holiday):
        batch_size = seq24.size(0) #tamaño del batch
        out = seq24 #reasignar datos de entrada
        block_idx = 0

        for i in range(self.num_layers):
            gru = self.blocks[block_idx]

            #inicializar celda de memoria y estado oculto aleatoriamente
            h0 = torch.zeros(1, batch_size, self.neurons_per_layer[i]).to(seq24.device)
            
            # out: all_hidde_states
            out, _ = gru(out, h0)

            #actualizar indice para saltar dropout
            block_idx += 1

            # Aplicar dropout si corresponde
            if block_idx < len(self.blocks) and isinstance(self.blocks[block_idx], nn.Dropout):
                out = self.blocks[block_idx](out)
                block_idx += 1

        last_hidden_size = out[:, -1, :]  # Último paso temporal
        output_month = self.embedding_mes(month).squeeze(1)
        output_weekday = self.embedding_weekday(weekday).squeeze(1)

        final_output = torch.cat((last_hidden_size, output_month, output_weekday, holiday), dim=1)

        return self.fc(final_output)

