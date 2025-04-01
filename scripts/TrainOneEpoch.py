import time
from tqdm import tqdm
def train_one_epoch(
                model,
                dataloader,
                criterion,
                optimizer,
                metrics,
                device,
                epoch = None
):
        """
        Ejecuta una epoch completa de entrenamiento.

        Args:
                model (nn.Module): Modelo a entrenar.
                dataloader (DataLoader): Dataloader con datos de entrenamiento.
                criterion (nn.Module): Función de pérdida.
                optimizer (Optimizer): Optimizador para actualizar pesos.
                metrics (torchmetrics.MetricCollection): Métricas de evaluación.
                device (str): "cpu" o "cuda".
                epoch (int, optional): Número de la época actual (solo para mostrar).

        Returns:
                dict: Diccionario con loss promedio y métricas acumuladas.
        """
        
        model.train() #poner modelo en modo entrenamiento

        #inicializar loss y nº de batches
        running_loss = 0.0
        num_batches = len(dataloader)

        #inicializar métricas y moverlas al device
        metrics.to(device)
        metrics.reset()

        #coger tiempo actual de referencia
        start_time = time.time()

        #inicializar barra de progreso
        #progress_bar envuelve el dataloader y lo monitoriza
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)


        for x_batch, y_batch in progress_bar:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad() #1. resetear gradientes

                y_pred = model(x_batch) #2. Forward

                loss = criterion(y_pred, y_batch) # 3. calcular loss
                running_loss += loss.item()
                
                batch_idx = progress_bar.n #indice del batch
                avg_loss_so_far = running_loss / (batch_idx + 1)

                #Mostrar en tiempo real el loss
                progress_bar.set_postfix({"Loss": avg_loss_so_far})


                loss.backward() #4. Propagar loss
                optimizer.step() #5. Actualizar pesos


                metrics.update(y_pred, y_batch)
        
        #calcular loss promedio del batch
        avg_loss = running_loss/num_batches

        #calcular metrica total: promedio de metricas de cada batch
        metric_results = metrics.compute()

        #crear diccionario de resutaldos: primer elemento del diccionario es el loss
        results = {"train_loss": avg_loss}

        #añadir elementos al diccionario: cada par clave-valor es una métrica
        results.update({k: v.item() for k,v in metric_results.items()})

        return results